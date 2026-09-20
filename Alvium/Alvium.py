"""
Allied Vision Alvium Camera Driver for Node Assistant.

Low-level interface for Alvium industrial cameras using the vmbpy SDK.
"""

from loguru import logger
import numpy as np
import cv2
import queue
import time

try:
    import vmbpy
except Exception as e:
    logger.warning(f"vmbpy is not installed or could not be loaded: {e}. Please install Vimba/VmbPy SDK to use Alvium cameras.")
    vmbpy = None


class Alvium_cam:
    """
    Low-level driver for Allied Vision Alvium cameras using vmbpy.
    """
    
    def __init__(self):
        self.defined = False
        self.vmb = None
        self.cam = None
        self.ranges = {
            'ExposureTime': (1.0, 10000000.0), # 1us to 10s
            'Gain': (0.0, 24.0)
        }
        self.processing_mode = "RGB8 (ISP)"
        
        self.frame_queue = queue.Queue(maxsize=4)
        self.had_error = False
        
    def init(self):
        self.had_error = False
        if self.defined:
            return True
            
        if vmbpy is None:
            logger.error("Alvium initialization failed: vmbpy is not available.")
            self.had_error = True
            return False
            
        try:
            self.vmb = vmbpy.VmbSystem.get_instance()
            self.vmb.__enter__()

            self.cam = self.vmb.get_all_cameras()[0]
            self.cam.__enter__()
            
            # Allow camera access and set up the default features
            self._setup_camera()
            
            # Brief pause before warmup
            time.sleep(0.2)
            
            try:
                self.cam.start_streaming(self._frame_callback)
                time.sleep(0.5)
                self.cam.stop_streaming()
            except Exception as e:
                logger.warning(f"Dry-run stream failed: {e}")
                
            time.sleep(0.5)
            # Ensure it is definitely in Video mode
            self.set_trigger_mode('Video')
            
            self.defined = True
            return True
        except Exception as e:
            logger.error(f"Alvium initialization failed: {e}")
            self.had_error = True
            if self.cam:
                self.cam.__exit__(None, None, None)
                self.cam = None
            if self.vmb:
                self.vmb.__exit__(None, None, None)
                self.vmb = None
            return False
            
    def _setup_camera(self):
        try:
            # Set the default pixel format according to processing mode
            self.set_format(self.processing_mode, restart_stream=False)
            
            # Ensure continuous acquisition mode
            try:
                self.cam.AcquisitionMode.set('Continuous')
            except: pass
            
            # RADICAL CLEANUP: Force all potential triggers OFF
            # Sometimes other selectors (like AcquisitionStart) are ON and block the flow
            for selector in ['FrameStart', 'AcquisitionStart', 'AcquisitionActive', 'ExposureStart']:
                try:
                    self.cam.TriggerSelector.set(selector)
                    self.cam.TriggerMode.set('Off')
                except: pass
            
            # Reset selector to FrameStart for future use
            try:
                self.cam.TriggerSelector.set('FrameStart')
            except: pass
            
            # UNLOCK FPS: Disable the fixed frame rate limit to allow ROI speedups
            try:
                self.cam.AcquisitionFrameRateEnable.set(False)
                logger.info("Frame rate limit disabled (Max performance mode)")
            except Exception as e:
                logger.debug(f"Could not disable FrameRate limit: {e}")
            
        except Exception as e:
            logger.warning(f"Error during radical camera cleanup: {e}")
            self.had_error = True
            
    def get_pixel_format(self):
        if not self.cam: return "Unknown"
        try:
            return str(self.cam.get_feature_by_name('PixelFormat').get())
        except: return "Unknown"
        
    def set_format(self, mode, restart_stream=True):
        """Set hardware pixel format and software processing mode"""
        self.processing_mode = mode
        if not self.cam: return
        try:
            was_streaming = False
            if restart_stream and getattr(self.cam, 'is_streaming', lambda: False)():
                was_streaming = True
                self.cam.stop_streaming()
                time.sleep(0.1)

            # Determine the hardware string
            if mode == "RGB8 (ISP)":
                hw_fmt = "RGB8"
            elif "Mono12" in mode:
                hw_fmt = "Mono12"
            elif "BayerGR12" in mode:
                hw_fmt = "BayerGR12"
            else:
                hw_fmt = "RGB8" # Fallback

            try:
                self.cam.get_feature_by_name('PixelFormat').set(hw_fmt)
                logger.info(f"Hardware PixelFormat set to {hw_fmt}")
            except Exception as e:
                logger.warning(f"Could not set hardware format to {hw_fmt}: {e}")
                self.had_error = True

            if was_streaming and restart_stream:
                self.start_streaming()
        except Exception as e:
            logger.error(f"Error handling format change: {e}")
            self.had_error = True

    def set_trigger_mode(self, mode):
        if not self.cam: return
        try:
            # Check if we need to selector FrameStart first
            try:
                self.cam.TriggerSelector.set('FrameStart')
            except: pass
            
            was_streaming = False
            try:
                if getattr(self.cam, 'is_streaming', lambda: False)():
                    was_streaming = True
                    self.cam.stop_streaming()
                    time.sleep(0.1) # Vital: give hardware time to settle
            except: pass
                
            if mode == "Video":
                self.cam.TriggerMode.set('Off')
            elif mode == "Trigger":
                self.cam.TriggerSource.set('Software')
                self.cam.TriggerMode.set('On')
                
            time.sleep(0.2) # Give more time for the hardware to switch states
            if was_streaming:
                try:
                    self.cam.start_streaming(self._frame_callback)
                except Exception as e:
                    logger.warning(f"Failed to restart stream after trigger change: {e}")
                    self.had_error = True
            
            logger.info(f"Trigger mode successfully switched to {mode}")
        except Exception as e:
            logger.error(f"Critical error switching trigger mode to {mode}: {e}")
            self.had_error = True
            # Try to recover the stream if it was lost
            try:
                if not self.cam.is_streaming():
                    self.cam.start_streaming(self._frame_callback)
            except: pass
            
    def trigger(self):
        if not self.cam: return
        try:
            try:
                self.cam.TriggerSelector.set('FrameStart')
            except: pass
            self.cam.TriggerSoftware.run()
        except Exception as e:
            logger.error(f"Software trigger failed: {e}")
            self.had_error = True
            
    def set_exposure(self, exposure_us):
        if not self.cam: return
        try:
            # Use cached range to avoid ultra-slow get_range() calls
            min_exp, max_exp = self.ranges.get('ExposureTime', (10.0, 1000000.0))
            val = max(min_exp, min(max_exp, float(exposure_us)))
            self.cam.ExposureTime.set(val)
        except Exception as e:
            logger.error(f"Exposure setting failed: {e}")
            self.had_error = True

    def auto_exposure(self):
        if not self.cam: return
        try:
            self.cam.ExposureAuto.set('Once')
            logger.info("Auto exposure triggered")
        except Exception as e:
            logger.error(f"Auto exposure failed: {e}")
            self.had_error = True

    def get_exposure(self):
        if not self.cam: return 2000.0
        try:
            return float(self.cam.ExposureTime.get())
        except: return 2000.0

    def set_gain(self, gain):
        if not self.cam: return
        try:
            # Use cached range to avoid ultra-slow get_range() calls
            min_gain, max_gain = self.ranges.get('Gain', (0.0, 24.0))
            val = max(min_gain, min(max_gain, float(gain)))
            self.cam.Gain.set(val)
        except Exception as e:
            logger.error(f"Gain setting failed: {e}")
            self.had_error = True

    def get_gain(self):
        if not self.cam: return 1.0
        try:
            return float(self.cam.Gain.get())
        except: return 1.0

    def set_binning(self, value):
        """Set hardware binning (Horizontal & Vertical)"""
        if not self.cam: return
        try:
            val = int(value)
            was_streaming = False
            if getattr(self.cam, 'is_streaming', lambda: False)():
                was_streaming = True
                self.cam.stop_streaming()
                time.sleep(0.1)

            # 1. Try to set binning mode to Average if supported (cleaner for colors)
            for axis in ['Horizontal', 'Vertical']:
                try:
                    mode_feat = getattr(self.cam, f'Binning{axis}Mode')
                    # 'Average' is usually better for noise, 'Sum' for low light
                    mode_feat.set('Average') 
                except: pass

            # 2. Set the binning values
            try:
                self.cam.BinningHorizontal.set(val)
                self.cam.BinningVertical.set(val)
                logger.info(f"Hardware Binning set to {val}x{val}")
                
                # 3. FORCE ROI RESET: Returning to lower binning often leaves the ROI cropped
                # We need to set Width and Height back to their max values
                try:
                    # Get max possible width/height for the current binning
                    w_max = self.cam.Width.get_range()[1]
                    h_max = self.cam.Height.get_range()[1]
                    self.cam.Width.set(w_max)
                    self.cam.Height.set(h_max)
                    logger.info(f"ROI Reset to max: {w_max}x{h_max}")
                except Exception as e:
                    logger.warning(f"Failed to reset ROI after binning: {e}")
                    
            except Exception as e:
                logger.error(f"Hardware binning failed: {e}")
                self.had_error = True

            if was_streaming:
                self.start_streaming()
        except Exception as e:
            logger.error(f"Error during hardware binning change: {e}")
            self.had_error = True

    def get_binning(self):
        if not self.cam: return 1
        try:
            return int(self.cam.BinningHorizontal.get())
        except: return 1

    def set_roi(self, width, height):
        """Set the Region of Interest (ROI) size"""
        if not self.cam: return
        try:
            was_streaming = False
            if getattr(self.cam, 'is_streaming', lambda: False)():
                was_streaming = True
                self.cam.stop_streaming()
                time.sleep(0.1)

            # Note: Width/Height must be increments of a certain value (usually 4 or 8)
            # We use the range and hope for the best, or the SDK handles it
            try:
                # Set offsets to 0 first to avoid out of bounds
                self.cam.OffsetX.set(0)
                self.cam.OffsetY.set(0)
                
                self.cam.Width.set(int(width))
                self.cam.Height.set(int(height))
                logger.info(f"ROI set to {width}x{height}")
            except Exception as e:
                logger.error(f"Failed to set ROI: {e}")
                self.had_error = True

            if was_streaming:
                self.start_streaming()
        except Exception as e:
            logger.error(f"Error during ROI change: {e}")
            self.had_error = True

    def get_roi(self):
        if not self.cam: return 0, 0
        try:
            return int(self.cam.Width.get()), int(self.cam.Height.get())
        except: return 0, 0

    def get_max_roi(self):
        if not self.cam: return 0, 0
        try:
            return self.cam.Width.get_range()[1], self.cam.Height.get_range()[1]
        except: return 0, 0
            
            
    def auto_white_balance(self):
        if not self.cam: return
        try:
            self.cam.BalanceWhiteAuto.set('Once')
            logger.info("Auto white balance triggered")
        except Exception as e:
            logger.error(f"Auto white balance failed: {e}")
            self.had_error = True
            
    def set_manual_white_balance(self, ratio_r, ratio_b):
        if not self.cam: return
        try:
            self.cam.BalanceWhiteAuto.set('Off')
            # Red
            self.cam.BalanceRatioSelector.set('Red')
            self.cam.BalanceRatio.set(float(ratio_r))
            # Blue
            self.cam.BalanceRatioSelector.set('Blue')
            self.cam.BalanceRatio.set(float(ratio_b))
        except Exception as e:
            logger.error(f"Manual white balance failed: {e}")
            self.had_error = True

    def get_white_balance(self):
        r, b = 1.0, 1.0
        if not self.cam: return r, b
        try:
            self.cam.BalanceRatioSelector.set('Red')
            r = float(self.cam.BalanceRatio.get())
            self.cam.BalanceRatioSelector.set('Blue')
            b = float(self.cam.BalanceRatio.get())
        except Exception:
            pass
        return r, b

    def start_streaming(self):
        if not self.cam: return
        for attempt in range(3):
            try:
                if getattr(self.cam, 'is_streaming', lambda: False)():
                    self.cam.stop_streaming()
                    time.sleep(0.1)
                self.cam.start_streaming(self._frame_callback)
                logger.debug("Camera stream successfully engaged.")
                return
            except Exception as e:
                logger.warning(f"Failed to start streaming (attempt {attempt+1}/3): {e}")
                time.sleep(0.5)
        logger.error("Camera refused to start streaming after 3 attempts.")
        self.had_error = True
        
    def stop_streaming(self):
        if not self.cam: return
        try:
            self.cam.stop_streaming()
        except:
            pass
            
    def _frame_callback(self, cam, stream, frame):
        try:
            if frame.get_status() == vmbpy.FrameStatus.Complete:
                try:
                    img = frame.as_numpy_ndarray()
                except Exception:
                    # Conversion needed for Packed formats (12p)
                    current_fmt = frame.get_pixel_format()
                    if "Bayer" in str(current_fmt):
                        img = frame.convert_pixel_format(vmbpy.PixelFormat.BayerGR16).as_numpy_ndarray()
                    else:
                        img = frame.convert_pixel_format(vmbpy.PixelFormat.Mono16).as_numpy_ndarray()

                mode = self.processing_mode
                
                # Squeeze the array if it has a useless 1-dimension (like vmbpy often returns: H, W, 1)
                if len(img.shape) == 3 and img.shape[2] == 1:
                    img = img.squeeze(-1)

                if mode == "RGB8 (ISP)":
                    if len(img.shape) == 3 and img.shape[2] >= 3:
                        # Convert RGB to BGR for OpenCV standard
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                elif "Mono12" in mode:
                    pass # already 1-channel uint16
                elif "BayerGR12" in mode:
                    if "Raw" in mode:
                        pass # keep mosaiced
                    else:
                        img = cv2.cvtColor(img, cv2.COLOR_BAYER_GB2BGR)
                        if "(Red)" in mode:
                            img = img[:, :, 2] # Extract Red (Index 2 in BGR)
                        elif "(Green)" in mode:
                            img = img[:, :, 1] # Extract Green (Index 1 in BGR)
                        elif "(Blue)" in mode:
                            img = img[:, :, 0] # Extract Blue (Index 0 in BGR)

                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put(img)
                
        except Exception as e:
            logger.error(f"Frame callback processing error: {e}")
            
        finally:
            cam.queue_frame(frame)

    def close(self):
        if self.cam:
            self.stop_streaming()
            self.cam.__exit__(None, None, None)
            self.cam = None
        if self.vmb:
            self.vmb.__exit__(None, None, None)
            self.vmb = None
        self.defined = False
