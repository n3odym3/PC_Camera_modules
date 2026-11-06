from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
from loguru import logger
import numpy as np

try:
    import pyvirtualcam
    PYVIRTUALCAM_AVAILABLE = True
except ImportError:
    pyvirtualcam = None
    PYVIRTUALCAM_AVAILABLE = False


class VirtualCameraEngine:
    """
    Core engine managing the virtual camera lifecycle using pyvirtualcam.
    Streams numpy image frames to an OS-level virtual camera device
    (e.g., OBS Virtual Camera, Unity Capture).
    """

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        backend: str = "obs",
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.backend = backend

        self.cam: Optional[Any] = None
        self.is_running: bool = False
        self.device_name: str = "None"
        self.last_error: str = ""

        # Statistics
        self.frames_sent: int = 0
        self.current_fps: float = 0.0
        self._fps_last_time: float = 0.0
        self._fps_frame_count: int = 0

    @staticmethod
    def is_library_available() -> bool:
        """Check if pyvirtualcam is installed."""
        return PYVIRTUALCAM_AVAILABLE

    @staticmethod
    def get_supported_backends() -> List[str]:
        """Return list of supported backend names."""
        return ["auto", "obs", "unitycapture", "v4l2loopback"]

    def start(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None,
        backend: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Start the virtual camera output stream.
        """
        if not PYVIRTUALCAM_AVAILABLE:
            self.last_error = "pyvirtualcam library is not installed. Please run: pip install pyvirtualcam"
            logger.error(self.last_error)
            return False, self.last_error

        if self.is_running:
            self.stop()

        if width is not None and width > 0:
            self.width = width
        if height is not None and height > 0:
            self.height = height
        if fps is not None and fps > 0:
            self.fps = fps
        if backend is not None:
            self.backend = backend

        # Map common aliases if needed
        backend_name = self.backend.lower().strip()
        if backend_name in ("unityvideo", "unity", "unity_capture"):
            backend_name = "unitycapture"
        elif backend_name in ("obs_virtualcam", "obsvirtualcam"):
            backend_name = "obs"

        backend_arg = None if backend_name == "auto" else backend_name

        try:
            kwargs: Dict[str, Any] = {
                "width": self.width,
                "height": self.height,
                "fps": self.fps,
                "fmt": pyvirtualcam.PixelFormat.BGR,
            }
            if backend_arg:
                kwargs["backend"] = backend_arg

            self.cam = pyvirtualcam.Camera(**kwargs)
            self.is_running = True
            self.device_name = str(getattr(self.cam, "device", "Virtual Camera"))
            self.frames_sent = 0
            self._fps_last_time = time.perf_counter()
            self._fps_frame_count = 0
            self.last_error = ""

            msg = f"Virtual Camera started: {self.device_name} ({self.width}x{self.height} @ {self.fps}fps)"
            logger.info(msg)
            return True, msg

        except Exception as e:
            self.is_running = False
            self.cam = None
            self.device_name = "None"
            err_str = str(e)
            if "unitycapture" in str(backend_arg) and not err_str:
                err_str = "Unity Capture device not found. Make sure Install.bat was run as Administrator."
            self.last_error = err_str if err_str else str(type(e).__name__)
            logger.error(f"Failed to start Virtual Camera: {self.last_error}")
            return False, self.last_error

    def send_frame(self, frame: Union[np.ndarray, Tuple[np.ndarray, str]]) -> bool:
        """
        Send a single frame to the virtual camera device.
        Automatically converts color formats (Grayscale/BGRA/RGBA to BGR)
        and resizes if dimensions do not match.
        """
        if not self.is_running or self.cam is None:
            return False

        img = frame[0] if isinstance(frame, tuple) else frame
        if img is None or not isinstance(img, np.ndarray):
            return False

        try:
            proc_img = img

            # 1. Resize first while channel count is minimal
            h, w = proc_img.shape[:2]
            if w != self.width or h != self.height:
                proc_img = cv2.resize(proc_img, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

            # 2. Ensure 8-bit depth before expanding channels
            if proc_img.dtype != np.uint8:
                proc_img = cv2.normalize(proc_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            # 3. Convert to 3-channel BGR
            if proc_img.ndim == 2:
                proc_img = cv2.cvtColor(proc_img, cv2.COLOR_GRAY2BGR)
            elif proc_img.ndim == 3:
                channels = proc_img.shape[2]
                if channels == 4:
                    proc_img = cv2.cvtColor(proc_img, cv2.COLOR_BGRA2BGR)
                elif channels == 1:
                    proc_img = cv2.cvtColor(proc_img, cv2.COLOR_GRAY2BGR)
                elif channels != 3:
                    return False
            else:
                return False

            if not proc_img.flags['C_CONTIGUOUS']:
                proc_img = np.ascontiguousarray(proc_img)

            self.cam.send(proc_img)
            self.frames_sent += 1
            self._fps_frame_count += 1

            # Update FPS calculation
            now = time.perf_counter()
            elapsed = now - self._fps_last_time
            if elapsed >= 1.0:
                self.current_fps = round(self._fps_frame_count / elapsed, 1)
                self._fps_frame_count = 0
                self._fps_last_time = now

            return True

        except Exception as e:
            self.last_error = str(e)
            logger.warning(f"Error sending frame to Virtual Camera: {e}")
            return False

    def stop(self) -> None:
        """
        Stop and close the virtual camera.
        """
        if self.cam is not None:
            try:
                self.cam.close()
            except Exception as e:
                logger.warning(f"Error closing virtual camera: {e}")
            self.cam = None

        self.is_running = False
        self.device_name = "None"
        self.current_fps = 0.0
        logger.info("Virtual Camera stopped.")
