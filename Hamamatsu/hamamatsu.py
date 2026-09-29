"""
Hamamatsu Camera Driver for Node Assistant.
Low-level wrapper for Hamamatsu cameras using pyDCAM.
"""
from typing import Optional, Tuple, Any
import numpy as np
import cv2
import time
from loguru import logger

try:
    from pyDCAM import *
    HAS_PYDCAM = True
except ImportError:
    HAS_PYDCAM = False

if not HAS_PYDCAM:
    class DummyEnum:
        def __getattr__(self, name: str) -> int:
            return 0
    DCAMPROPMODEVALUE = DummyEnum()
    DCAMIDPROP = DummyEnum()


class Hamamatsu:
    """
    Low-level driver for Hamamatsu cameras using pyDCAM.
    """
    
    def __init__(self) -> None:
        self.defined = False
        self.trigger_mode = False
        self.imgcount = 0
        self.width = 2048
        self.height = 2048
        self.Camera: Any = None
        self.hwait: Any = None
        self.buffer_count = 16
        self.readout_speed = "Fastest"

    def init(self) -> bool:
        """Initialize the camera API, open camera 0, allocate buffers, and start capture."""
        if not HAS_PYDCAM:
            logger.error("Hamamatsu initialization failed: pyDCAM library not found.")
            return False

        if self.defined:
            return True

        try:
            dcamapi_init()
            self.Camera = HDCAM(0)
            
            # Set readout speed
            if self.readout_speed == "Fastest":
                self.Camera.readout_speed = DCAMPROPMODEVALUE.DCAMPROP_READOUTSPEED__FASTEST
            else:
                self.Camera.readout_speed = DCAMPROPMODEVALUE.DCAMPROP_READOUTSPEED__SLOWEST
                
            self.Camera.exposure_time = 0.05
            self.Camera.dcambuf_alloc(self.buffer_count)
            self.hwait = self.Camera.dcamwait_open()
            self.Camera.dcamcap_start()

            # Attempt to determine width and height from first frame or DCAM properties
            try:
                # Query DCAM image width and height properties
                self.width = int(self.Camera.dcamprop_getvalue(DCAMIDPROP.DCAM_IDPROP_IMAGE_WIDTH)[0])
                self.height = int(self.Camera.dcamprop_getvalue(DCAMIDPROP.DCAM_IDPROP_IMAGE_HEIGHT)[0])
            except Exception:
                # Try getting a frame to extract size
                try:
                    array = self.Camera.dcambuf_copyframe()
                    frame = np.array(array)
                    self.width = frame.shape[1]
                    self.height = frame.shape[0]
                except Exception:
                    # Keep defaults if queries fail
                    pass

            self.defined = True
            logger.success(f"Hamamatsu camera initialized successfully ({self.width}x{self.height})")
            return True
        except Exception as e:
            logger.error(f"Hamamatsu initialization failed: {e}")
            self.close()
            return False

    def set_exposure(self, exposure_us: float) -> None:
        """Set exposure time (receives microseconds, converts to seconds for HDCAM)."""
        if not self.defined or self.Camera is None:
            return
        try:
            exposure_sec = exposure_us / 1000000.0
            self.Camera.exposure_time = exposure_sec
            logger.debug(f"Hamamatsu exposure set to {exposure_sec}s ({exposure_us}µs)")
        except Exception as e:
            logger.error(f"Failed to set exposure: {e}")

    def set_trigger(self, mode: str) -> None:
        """Set trigger source mode: 'Video' (Internal), 'Trigger' (Hardware/External) or 'Trigger Software' (Software)."""
        if not self.defined or self.Camera is None:
            return
        try:
            logger.info(f"Stopping capture to change Hamamatsu trigger mode to: {mode}")
            self.Camera.dcamcap_stop()
            try:
                self.Camera.dcambuf_release()
                logger.debug("Hamamatsu buffers released for property configuration")
            except Exception as release_err:
                logger.debug(f"Precautionary buffer release ignored/failed: {release_err}")

            match mode:
                case "Video":
                    # Disable global exposure
                    try:
                        none_val = getattr(DCAMPROPMODEVALUE, 'DCAMPROP_TRIGGER_GLOBALEXPOSURE__NONE', 1)
                        self.Camera.dcamprop_setvalue(
                            DCAMIDPROP.DCAM_IDPROP_TRIGGER_GLOBALEXPOSURE,
                            none_val
                        )
                    except Exception as e:
                        logger.debug(f"Could not reset global exposure to NONE: {e}")

                    self.Camera.dcamprop_setvalue(
                        DCAMIDPROP.DCAM_IDPROP_TRIGGERSOURCE,
                        DCAMPROPMODEVALUE.DCAMPROP_TRIGGERSOURCE__INTERNAL
                    )
                    self.trigger_mode = False
                    logger.info("Hamamatsu set to internal Video mode")

                case "Trigger":
                    self.Camera.dcamprop_setvalue(
                        DCAMIDPROP.DCAM_IDPROP_TRIGGERSOURCE,
                        DCAMPROPMODEVALUE.DCAMPROP_TRIGGERSOURCE__EXTERNAL
                    )
                    try:
                        self.Camera.dcamprop_setvalue(
                            DCAMIDPROP.DCAM_IDPROP_TRIGGERPOLARITY,
                            DCAMPROPMODEVALUE.DCAMPROP_TRIGGERPOLARITY__POSITIVE
                        )
                    except Exception as polarity_err:
                        logger.warning(f"Could not set Hamamatsu trigger polarity to Positive: {polarity_err}")

                    try:
                        self.Camera.dcamprop_setvalue(
                            DCAMIDPROP.DCAM_IDPROP_TRIGGERACTIVE,
                            DCAMPROPMODEVALUE.DCAMPROP_TRIGGERACTIVE__EDGE
                        )
                        logger.info("Hamamatsu set to Edge trigger active mode")
                    except Exception as active_err:
                        logger.warning(f"Could not set Hamamatsu trigger active to Edge: {active_err}")

                    # try:
                    #     self.Camera.dcamprop_setvalue(
                    #         DCAMIDPROP.DCAM_IDPROP_TRIGGERACTIVE,
                    #         DCAMPROPMODEVALUE.DCAMPROP_TRIGGERACTIVE__LEVEL
                    #     )
                    #     logger.info("Hamamatsu set to Level trigger active mode")
                    # except Exception as active_err:
                    #     logger.warning(f"Could not set Hamamatsu trigger active to Level: {active_err}")

                    try:
                        global_exposure_val = getattr(DCAMPROPMODEVALUE, 'DCAMPROP_TRIGGER_GLOBALEXPOSURE__GLOBALRESET', 5)
                        self.Camera.dcamprop_setvalue(
                            DCAMIDPROP.DCAM_IDPROP_TRIGGER_GLOBALEXPOSURE,
                            global_exposure_val
                        )
                        logger.info("Hamamatsu set to Global Reset trigger global exposure mode")
                    except Exception as global_exp_err:
                        err_str = str(global_exp_err)
                        if "DCAMERR_NOTSTABLE" in err_str:
                            logger.warning("Could not set Hamamatsu trigger global exposure to Global Reset (DCAMERR_NOTSTABLE). Exposure time might be too short relative to readout time. Try increasing exposure time first.")
                        else:
                            logger.warning(f"Could not set Hamamatsu trigger global exposure to Global Reset: {global_exp_err}")
                    
                    self.trigger_mode = True
                    logger.info("Hamamatsu set to hardware Trigger (External) mode with Rising Edge polarity, Edge Active and Global Reset active")
                
                case "Trigger Software":
                    # Disable global exposure
                    try:
                        none_val = getattr(DCAMPROPMODEVALUE, 'DCAMPROP_TRIGGER_GLOBALEXPOSURE__NONE', 1)
                        self.Camera.dcamprop_setvalue(
                            DCAMIDPROP.DCAM_IDPROP_TRIGGER_GLOBALEXPOSURE,
                            none_val
                        )
                    except Exception as e:
                        logger.debug(f"Could not reset global exposure to NONE: {e}")

                    self.Camera.dcamprop_setvalue(
                        DCAMIDPROP.DCAM_IDPROP_TRIGGERSOURCE,
                        DCAMPROPMODEVALUE.DCAMPROP_TRIGGERSOURCE__SOFTWARE
                    )
                    self.trigger_mode = True
                    logger.info("Hamamatsu set to software Trigger mode")

            # Reallocate buffer and restart capture
            self.Camera.dcambuf_alloc(self.buffer_count)
            self.Camera.dcamcap_start()
            logger.info("Hamamatsu capture restarted successfully")
        except Exception as e:
            logger.error(f"Failed to set Hamamatsu trigger mode: {e}")

    def trigger(self) -> None:
        """Fire a software trigger."""
        if not self.defined or self.Camera is None:
            return
        try:
            self.Camera.dcamcap_firetrigger()
        except Exception as e:
            err_str = str(e)
            if "DCAMERR_NOTSUPPORT" in err_str:
                logger.warning("Software trigger is not supported in the current trigger mode (must be in Trigger Software mode).")
            else:
                logger.error(f"Failed to fire Hamamatsu software trigger: {e}")

    def set_aoi(self, width: int, height: int, x: int = 0, y: int = 0) -> Any:
        """Set Area of Interest (subarray). Returns actual (width, height) or False on failure."""
        if not self.defined:
            self.width = width
            self.height = height
            return True
        try:
            self.Camera.dcamcap_stop()
            # Turn subarray mode ON (2 corresponds to DCAM_PROP_MODE__ON)
            self.Camera.dcamprop_setvalue(DCAMIDPROP.DCAM_IDPROP_SUBARRAYMODE, 2)
            self.Camera.dcamprop_setvalue(DCAMIDPROP.DCAM_IDPROP_SUBARRAY_HPOS, x)
            self.Camera.dcamprop_setvalue(DCAMIDPROP.DCAM_IDPROP_SUBARRAY_VPOS, y)
            self.Camera.dcamprop_setvalue(DCAMIDPROP.DCAM_IDPROP_SUBARRAY_HSIZE, width)
            self.Camera.dcamprop_setvalue(DCAMIDPROP.DCAM_IDPROP_SUBARRAY_VSIZE, height)
            
            # Reallocate buffer if necessary, depending on pyDCAM logic
            try:
                self.Camera.dcambuf_release()
            except Exception:
                pass
            self.Camera.dcambuf_alloc(self.buffer_count)
            
            self.Camera.dcamcap_start()
            self.width = width
            self.height = height
            logger.info(f"Hamamatsu AOI set to {width}x{height} at ({x}, {y})")
            return (self.width, self.height)
        except Exception as e:
            logger.error(f"Failed to set Hamamatsu AOI: {e}")
            return False

    def set_binning(self, binning_factor: int) -> Tuple[int, int]:
        """Set binning (1, 2, 4). Returns new (width, height)."""
        if not self.defined or self.Camera is None:
            return (self.width, self.height)
        try:
            logger.info(f"Stopping capture to change Hamamatsu binning to: {binning_factor}x{binning_factor}")
            self.Camera.dcamcap_stop()
            try:
                self.Camera.dcambuf_release()
            except Exception as release_err:
                logger.debug(f"Precautionary buffer release ignored/failed: {release_err}")

            match binning_factor:
                case 1:
                    val = DCAMPROPMODEVALUE.DCAMPROP_BINNING__1
                case 2:
                    val = DCAMPROPMODEVALUE.DCAMPROP_BINNING__2
                case 4:
                    val = DCAMPROPMODEVALUE.DCAMPROP_BINNING__4
                case _:
                    val = DCAMPROPMODEVALUE.DCAMPROP_BINNING__1
                    binning_factor = 1

            self.Camera.dcamprop_setvalue(
                DCAMIDPROP.DCAM_IDPROP_BINNING,
                val
            )

            # Reallocate buffer and restart capture
            self.Camera.dcambuf_alloc(self.buffer_count)
            self.Camera.dcamcap_start()
            
            # Query updated width and height
            try:
                self.width = int(self.Camera.dcamprop_getvalue(DCAMIDPROP.DCAM_IDPROP_IMAGE_WIDTH)[0])
                self.height = int(self.Camera.dcamprop_getvalue(DCAMIDPROP.DCAM_IDPROP_IMAGE_HEIGHT)[0])
            except Exception as size_err:
                logger.warning(f"Could not query size after binning change, using default division: {size_err}")
                self.width = 2048 // binning_factor
                self.height = 2048 // binning_factor

            logger.info(f"Hamamatsu binning set to {binning_factor}x{binning_factor} ({self.width}x{self.height})")
            return (self.width, self.height)
        except Exception as e:
            logger.error(f"Failed to set Hamamatsu binning: {e}")
            try:
                self.Camera.dcambuf_alloc(self.buffer_count)
                self.Camera.dcamcap_start()
            except Exception:
                pass
            return (self.width, self.height)

    def set_readout_speed(self, speed: str) -> None:
        """Set readout speed: 'Fastest' or 'Slowest'."""
        self.readout_speed = speed
        if not self.defined or self.Camera is None:
            return
        try:
            logger.info(f"Stopping capture to change Hamamatsu readout speed to: {speed}")
            self.Camera.dcamcap_stop()
            try:
                self.Camera.dcambuf_release()
            except Exception as release_err:
                logger.debug(f"Precautionary buffer release failed: {release_err}")

            if speed == "Fastest":
                self.Camera.readout_speed = DCAMPROPMODEVALUE.DCAMPROP_READOUTSPEED__FASTEST
            else:
                self.Camera.readout_speed = DCAMPROPMODEVALUE.DCAMPROP_READOUTSPEED__SLOWEST

            # Reallocate buffer and restart capture
            self.Camera.dcambuf_alloc(self.buffer_count)
            self.Camera.dcamcap_start()
            logger.info(f"Hamamatsu readout speed set to {speed}")
        except Exception as e:
            logger.error(f"Failed to set Hamamatsu readout speed: {e}")
            try:
                self.Camera.dcambuf_alloc(self.buffer_count)
                self.Camera.dcamcap_start()
            except Exception:
                pass

    def close(self) -> None:
        """Stop capture, release buffers, and uninitialize dcam api."""
        if self.Camera is not None:
            try:
                self.Camera.dcamcap_stop()
                self.Camera.dcambuf_release()
            except Exception:
                pass
            self.Camera = None

        if HAS_PYDCAM:
            try:
                dcamapi_uninit()
            except Exception:
                pass

        self.defined = False
        logger.info("Hamamatsu camera connection closed")

    def check_event(self, timeout_ms: int = 10) -> bool:
        """Check for a wait event from the camera wait handle."""
        if not self.defined or self.hwait is None:
            return False
        try:
            ret = self.hwait.dcamwait_start(timeout=timeout_ms)
            return bool(ret)
        except Exception:
            return False

    def get_image(self) -> Optional[np.ndarray]:
        """Capture and return the high-depth image frame (usually 16-bit)."""
        if not self.defined or self.Camera is None:
            return None
        try:
            array = self.Camera.dcambuf_copyframe()
            frame = np.array(array)
            return frame
        except Exception as e:
            logger.debug(f"Hamamatsu frame capture transient error: {e}")
            return None

    def save_img(self, frame: np.ndarray) -> None:
        """Save a frame to disk as PNG."""
        try:
            cv2.imwrite(f"IMG{self.imgcount}.png", frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            self.imgcount += 1
        except Exception as e:
            logger.error(f"Failed to save image: {e}")


hamamatsu = Hamamatsu()
