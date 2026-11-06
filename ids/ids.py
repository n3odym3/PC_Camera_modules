"""
IDS Camera Driver for Node Assistant.

Low-level interface for IDS industrial cameras using the pyueye SDK.
"""

from pyueye import ueye
import cv2
import numpy as np
from loguru import logger
import threading


class IDS:
    """
    Low-level driver for IDS industrial cameras.
    
    Provides camera initialization, configuration, and frame capture
    using the pyueye SDK.
    """
    
    def __init__(self):
        self.defined = False
        self.Camera = ueye.HIDS(0)
        self.sensorInfo = ueye.SENSORINFO()
        self.camInfo = ueye.CAMINFO()
        self.ImageMemory = ueye.c_mem_p()
        self.MemoryID = ueye.int()

        self.buffers = []
        self.buffer_count = 1

        self.pitch = ueye.INT()
        self.pClock = ueye.INT()
        self.sClock = ueye.INT()
        self.width = 1280
        self.height = 1024
        self.ColorMode = ueye.IS_CM_MONO12
        self.BitsPerPixel = ueye.INT(16)
        self.BytesPerPixel = int(self.BitsPerPixel / 8)
        self._pixels_per_line = 0
        self.trigger_mode = False
        self.imgcount = 0

        self.signaled = None
        self.eventcount = None
        self._event_id_uint = None
        self._event_id_size = 0
        self._wait_param = None
        self._wait_param_size = 0

    def _allocate_buffers(self) -> bool:
        """Allocate image memory sequence and inquire pitch."""
        self.buffers = []
        for i in range(self.buffer_count):
            mem_ptr = ueye.c_mem_p()
            mem_id = ueye.int()
            ret_alloc = ueye.is_AllocImageMem(
                self.Camera, self.width, self.height, self.BitsPerPixel, mem_ptr, mem_id
            )
            if ret_alloc != ueye.IS_SUCCESS:
                logger.error(f"Buffer {i} allocation error: code {ret_alloc}")
                continue
            ueye.is_AddToSequence(self.Camera, mem_ptr, mem_id)
            self.buffers.append((mem_ptr, mem_id))

        if not self.buffers:
            logger.error("Buffer selection failed")
            return False

        self.ImageMemory, self.MemoryID = self.buffers[0]
        ueye.is_SetImageMem(self.Camera, self.ImageMemory, self.MemoryID)
        ueye.is_InquireImageMem(
            self.Camera,
            self.ImageMemory,
            self.MemoryID,
            ueye.INT(self.width),
            ueye.INT(self.height),
            self.BitsPerPixel,
            self.pitch,
        )
        self._pixels_per_line = int(self.pitch.value) // self.BytesPerPixel
        return True

    def init(self):
        """Initialize the camera and allocate buffers"""

        if not self.defined:
            ret = ueye.is_InitCamera(self.Camera, None)
            if ret != ueye.IS_SUCCESS:
                return False

            ret = ueye.is_GetCameraInfo(self.Camera, self.camInfo)
            ret = ueye.is_GetSensorInfo(self.Camera, self.sensorInfo)

            ret = ueye.is_ResetToDefault(self.Camera)
            ret = ueye.is_SetDisplayMode(self.Camera, ueye.IS_SET_DM_DIB)

            self.rectAOI = ueye.IS_RECT()
            ret = ueye.is_AOI(self.Camera, ueye.IS_AOI_IMAGE_GET_AOI, self.rectAOI, ueye.sizeof(self.rectAOI))
            self.width = int(self.rectAOI.s32Width)
            self.height = int(self.rectAOI.s32Height)

            if not self._allocate_buffers():
                return False

            ret = ueye.is_SetColorMode(self.Camera, self.ColorMode)
            ret = ueye.is_DeviceFeature(self.Camera, ueye.IS_DEVICE_FEATURE_CAP_SHUTTER_MODE_ROLLING_GLOBAL_START, ueye.bool(1), ueye.sizeof(ueye.bool(1)))
            
            self.set_pixel_clock(86)
            self.set_gain(0)
            self.set_shutter("Global")
            self.set_trigger("Video")

            self.set_frame_event()
            self.defined = True
        return True

    def set_shutter(self, shutter="Global"):
        """Set the shutter mode"""

        shutter_type = ueye.IS_DEVICE_FEATURE_CAP_SHUTTER_MODE_GLOBAL
        match shutter:
            case "Global":
                shutter_type = ueye.IS_DEVICE_FEATURE_CAP_SHUTTER_MODE_GLOBAL
            case "Global alt.":
                shutter_type = ueye.IS_DEVICE_FEATURE_CAP_SHUTTER_MODE_GLOBAL_ALTERNATIVE_TIMING
            case "Rolling":
                shutter_type = ueye.IS_DEVICE_FEATURE_CAP_SHUTTER_MODE_ROLLING
            case "Rolling glob.":
                shutter_type = ueye.IS_DEVICE_FEATURE_CAP_SHUTTER_MODE_ROLLING_GLOBAL_START
                
        ret = ueye.is_DeviceFeature(self.Camera, ueye.IS_DEVICE_FEATURE_CMD_SET_SHUTTER_MODE, ueye.INT(shutter_type), ueye.sizeof(ueye.INT()))
        ret = ueye.is_CaptureVideo(self.Camera, ueye.IS_DONT_WAIT)
    
    def set_black_level_auto(self, enabled=True):
        """Enable or disable automatic black level correction"""

        if enabled:
            ueye.is_Blacklevel(self.Camera, ueye.IS_BLACKLEVEL_CMD_SET_MODE, ueye.INT(ueye.IS_AUTO_BLACKLEVEL_ON), ueye.sizeof(ueye.INT()))
        else:
            ueye.is_Blacklevel(self.Camera, ueye.IS_BLACKLEVEL_CMD_SET_MODE, ueye.INT(ueye.IS_AUTO_BLACKLEVEL_OFF), ueye.sizeof(ueye.INT()))
    
    def set_black_offset(self, offset):
        """Set the black level offset"""

        ueye.is_Blacklevel(self.Camera, ueye.IS_BLACKLEVEL_CMD_SET_OFFSET, ueye.INT(offset), ueye.sizeof(ueye.INT()))

    def set_pixel_clock(self, clock):
        """Set the pixel clock in MHz"""

        ueye_number = ueye.UINT()
        ueye.is_PixelClock(self.Camera, ueye.IS_PIXELCLOCK_CMD_SET, ueye.UINT(clock), ueye.sizeof(ueye_number))

    def set_gain(self, gain):
        """Set the hardware gain (0-100)"""

        ueye.is_SetHardwareGain(self.Camera, ueye.INT(gain), ueye.IS_IGNORE_PARAMETER, ueye.IS_IGNORE_PARAMETER, ueye.IS_IGNORE_PARAMETER)

    def set_exposure(self, exposure):
        """Set the exposure time in microseconds"""

        expos = ueye.DOUBLE(exposure / 1000)
        ueye.is_Exposure(self.Camera, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, expos, ueye.UINT(8))

    def get_trigger_debounce(self):
        """Get and print trigger debounce settings"""
        mode = ueye.UINT()
        delay = ueye.UINT()
        min_us = ueye.UINT()
        max_us = ueye.UINT()
        inc_us = ueye.UINT()

        ueye.is_TriggerDebounce(self.Camera, ueye.TRIGGER_DEBOUNCE_CMD_GET_MODE, mode, ueye.sizeof(mode))
        ueye.is_TriggerDebounce(self.Camera, ueye.TRIGGER_DEBOUNCE_CMD_GET_DELAY_TIME, delay, ueye.sizeof(delay))
        ueye.is_TriggerDebounce(self.Camera, ueye.TRIGGER_DEBOUNCE_CMD_GET_DELAY_TIME_MIN, min_us, ueye.sizeof(min_us))
        ueye.is_TriggerDebounce(self.Camera, ueye.TRIGGER_DEBOUNCE_CMD_GET_DELAY_TIME_MAX, max_us, ueye.sizeof(max_us))
        ueye.is_TriggerDebounce(self.Camera, ueye.TRIGGER_DEBOUNCE_CMD_GET_DELAY_TIME_INC, inc_us, ueye.sizeof(inc_us))

        logger.info(f"Trigger Debounce - Mode: {mode.value}, Delay: {delay.value}µs, Range: {min_us.value}-{max_us.value}µs")

    def set_trigger_debounce(self, mode="AUTO", delay_us=50):
        """Set trigger debounce mode and delay"""

        mode_map = {
            "NONE": ueye.TRIGGER_DEBOUNCE_MODE_NONE,
            "RISING": ueye.TRIGGER_DEBOUNCE_MODE_RISING_EDGE,
            "FALLING": ueye.TRIGGER_DEBOUNCE_MODE_FALLING_EDGE,
            "BOTH": ueye.TRIGGER_DEBOUNCE_MODE_BOTH_EDGES,
            "AUTO": ueye.TRIGGER_DEBOUNCE_MODE_AUTOMATIC,
        }

        mode_const = mode_map.get(mode.upper(), ueye.TRIGGER_DEBOUNCE_MODE_AUTOMATIC)

        ueye.is_TriggerDebounce(self.Camera, ueye.TRIGGER_DEBOUNCE_CMD_SET_MODE,
                                ueye.UINT(mode_const), ueye.sizeof(ueye.UINT()))
        ueye.is_TriggerDebounce(self.Camera, ueye.TRIGGER_DEBOUNCE_CMD_SET_DELAY_TIME,
                                ueye.UINT(delay_us), ueye.sizeof(ueye.UINT()))

        logger.info(f"Trigger debounce set to mode '{mode}' with delay {delay_us}µs")

    def set_trigger(self, mode):
        """Set trigger mode (Video or Trigger)"""

        match mode:
            case "Video":
                ueye.is_SetExternalTrigger(self.Camera, ueye.IS_SET_TRIGGER_OFF)
                ueye.is_CaptureVideo(self.Camera, ueye.IS_DONT_WAIT)
                self.trigger_mode = False
            case "Trigger":
                self.get_trigger_debounce()
                ueye.is_SetExternalTrigger(self.Camera, ueye.IS_SET_TRIGGER_LO_HI)
                ueye.is_CaptureVideo(self.Camera, ueye.IS_DONT_WAIT)
                self.trigger_mode = True

    def trigger(self):
        """Force a software trigger"""

        if self.defined:
            ueye.is_ForceTrigger(self.Camera)

    def set_aoi(self, width, height, x=0, y=0):
        """
        Set the Area of Interest (AOI) / Resolution.
        
        Note: This reallocates image memory.
        """
        if not self.defined:
            # If not initialized yet, just store values
            self.width = width
            self.height = height
            return True
            
        # Stop capture
        ueye.is_StopLiveVideo(self.Camera, ueye.IS_WAIT)
        
        # Free existing memory
        for mem_ptr, mem_id in self.buffers:
            ueye.is_FreeImageMem(self.Camera, mem_ptr, mem_id)
        self.buffers = []
        
        # Set AOI
        rectAOI = ueye.IS_RECT()
        rectAOI.s32X = ueye.int(x)
        rectAOI.s32Y = ueye.int(y)
        rectAOI.s32Width = ueye.int(width)
        rectAOI.s32Height = ueye.int(height)
        
        ret = ueye.is_AOI(self.Camera, ueye.IS_AOI_IMAGE_SET_AOI, rectAOI, ueye.sizeof(rectAOI))
        if ret != ueye.IS_SUCCESS:
            logger.error(f"Failed to set AOI: {ret}")
            return False
            
        # Update internal dims (get actual values back from camera)
        ueye.is_AOI(self.Camera, ueye.IS_AOI_IMAGE_GET_AOI, rectAOI, ueye.sizeof(rectAOI))
        self.width = int(rectAOI.s32Width)
        self.height = int(rectAOI.s32Height)
        
        # Re-allocate memory
        self._allocate_buffers()
        
        # Restart live video
        ueye.is_CaptureVideo(self.Camera, ueye.IS_DONT_WAIT)
        
        logger.info(f"Resolution set to {self.width}x{self.height} (pitch: {self.pitch.value})")
        return (self.width, self.height)

    def close(self):
        """Release camera resources"""
        if self.defined:
            for mem_ptr, mem_id in self.buffers:
                ueye.is_FreeImageMem(self.Camera, mem_ptr, mem_id)
            self.buffers = []
            ueye.is_ExitCamera(self.Camera)
            self.defined = False

    def get_image(self):
        """
        Capture an image from the camera    
        
        Returns:
            numpy.ndarray: 16-bit grayscale frame.
        """
        if not self.defined or not self.buffers:
            return None
            
        try:
            w = self.width
            h = self.height
            pixels_per_line = self._pixels_per_line or (int(self.pitch.value) // self.BytesPerPixel)

            array = ueye.get_data(self.ImageMemory, w, h, self.BitsPerPixel, self.pitch, copy=False)
            raw_data = np.frombuffer(array, dtype=np.uint16)
            
            frame = raw_data.reshape((h, pixels_per_line))
            if pixels_per_line != w:
                frame = frame[:, :w]
            return frame.copy()
        except Exception as e:
            logger.debug(f"IDS Buffer transient error: {e}")
            return None

    def set_frame_event(self):
        """Initialize the frame event for synchronization"""

        eventID = ueye.IS_SET_EVENT_FRAME
        self.signaled = ueye.UINT()
        self.eventcount = ueye.UINT()

        init_param = ueye.IS_INIT_EVENT(nEvent=eventID, bManualReset=True, bInitialState=False)
        ueye.is_Event(self.Camera, ueye.IS_EVENT_CMD_INIT, init_param, ueye.sizeof(init_param))

        self._event_id_uint = ueye.UINT(eventID)
        self._event_id_size = ueye.sizeof(self._event_id_uint)
        ueye.is_Event(self.Camera, ueye.IS_EVENT_CMD_ENABLE, self._event_id_uint, self._event_id_size)

        self._wait_param = ueye.IS_WAIT_EVENT(
            nEvent=eventID,
            nTimeoutMilliseconds=100,
            nSignaled=self.signaled,
            nSetCount=self.eventcount,
        )
        self._wait_param_size = ueye.sizeof(self._wait_param)

    def check_event(self, timeout_ms=100):
        """
        Check for a frame event from the camera.
        
        Args:
            timeout_ms: How long to wait for the event in ms.
                        Using a non-zero value avoids high CPU usage (busy-wait).
        
        Returns:
            bool: True if a frame event is detected, False otherwise.
        """
        wait_param = self._wait_param
        if wait_param is None:
            self.set_frame_event()
            wait_param = self._wait_param

        if wait_param.nTimeoutMilliseconds != timeout_ms:
            wait_param.nTimeoutMilliseconds = timeout_ms

        ret = ueye.is_Event(self.Camera, ueye.IS_EVENT_CMD_WAIT, wait_param, self._wait_param_size)

        if ret == ueye.IS_SUCCESS:
            ueye.is_Event(self.Camera, ueye.IS_EVENT_CMD_RESET, wait_param, self._wait_param_size)
            return True
        else:
            return False

    def save_img(self, frame):
        """Save a frame to disk as PNG"""

        cv2.imwrite(f"IMG{self.imgcount}.png", frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        self.imgcount += 1
