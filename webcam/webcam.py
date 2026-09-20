from typing import Any, Dict, List, Optional, Tuple
from loguru import logger
import cv2

try:
    from .pygrabber.dshow_graph import FilterGraph
    PYGRABBER_AVAILABLE = True
except ImportError:
    try:
        from pygrabber.dshow_graph import FilterGraph
        PYGRABBER_AVAILABLE = True
    except ImportError:
        PYGRABBER_AVAILABLE = False


class Webcam:
    """
    DirectShow-based video grabber and camera management interface.

    Handles device enumeration, hardware capability / resolution probing,
    theoretical framerate calculations, and frame acquisition via OpenCV DirectShow.
    """

    def __init__(self) -> None:
        """Initialize webcam device parameters and state."""
        self.cap: Optional[cv2.VideoCapture] = None
        self.running: bool = False
        self.width: int = 0
        self.height: int = 0
        self.theoretical_fps: float = 0.0
        self.cam_index: int = -1
        self.cam_name: str = ""
        self.available_formats: List[Dict[str, Any]] = []

    def list_devices(self) -> List[str]:
        """List all connected video input devices via DirectShow."""
        if PYGRABBER_AVAILABLE:
            try:
                return FilterGraph().get_input_devices()
            except Exception as e:
                logger.warning(f"DirectShow device enumeration error: {e}")
        
        # Fallback: test cv2.VideoCapture indices
        devices = []
        for i in range(5):
            temp_cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if temp_cap.isOpened():
                devices.append(f"Camera {i}")
                temp_cap.release()
        return devices

    def scan_resolutions_and_fps(self, cam_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Scans all hardware-supported resolutions and theoretical frame rates
        for the given camera device via DirectShow stream capabilities.
        """
        target_name = cam_name or self.cam_name
        if not target_name or not PYGRABBER_AVAILABLE:
            return self._fallback_resolutions()

        try:
            graph = FilterGraph()
            devices = graph.get_input_devices()
            if target_name not in devices:
                return self._fallback_resolutions()

            idx = devices.index(target_name)
            graph.add_video_input_device(idx)
            video_input = graph.get_input_device()
            raw_formats = video_input.get_formats()

            parsed: List[Dict[str, Any]] = []
            seen_combos = set()

            for item in raw_formats:
                w = int(item.get("width", 0))
                h = int(item.get("height", 0))
                if w <= 0 or h <= 0:
                    continue

                min_fps = float(item.get("min_framerate", 0.0))
                max_fps = float(item.get("max_framerate", 0.0))
                # DirectShow min_framerate / max_framerate calculation:
                # In dshow_graph: min_framerate = 10^7 / MinFrameInterval = theoretical MAX FPS
                # max_framerate = 10^7 / MaxFrameInterval = theoretical MIN FPS
                theoretical_max_fps = max(min_fps, max_fps)
                media_type = str(item.get("media_type_str", ""))

                combo_key = (w, h, round(theoretical_max_fps, 1), media_type)
                if combo_key not in seen_combos:
                    seen_combos.add(combo_key)
                    parsed.append({
                        "width": w,
                        "height": h,
                        "fps": round(theoretical_max_fps, 1),
                        "media_type": media_type,
                        "label": f"{w}x{h} @ {theoretical_max_fps:.0f} FPS ({media_type})" if media_type else f"{w}x{h} @ {theoretical_max_fps:.0f} FPS",
                        "short_label": f"{w}x{h} (max {theoretical_max_fps:.0f} FPS)",
                    })

            # Sort descending by resolution area, then by FPS
            parsed.sort(key=lambda x: (x["width"] * x["height"], x["fps"]), reverse=True)
            self.available_formats = parsed
            return parsed

        except Exception as e:
            logger.warning(f"Error scanning DirectShow capabilities for '{target_name}': {e}")
            return self._fallback_resolutions()

    def _fallback_resolutions(self) -> List[Dict[str, Any]]:
        """Fallback standard resolutions when DirectShow capability query is unavailable."""
        standard = [
            {"width": 3840, "height": 2160, "fps": 30.0, "media_type": "RGB", "label": "3840x2160 @ 30 FPS", "short_label": "3840x2160"},
            {"width": 1920, "height": 1080, "fps": 30.0, "media_type": "RGB", "label": "1920x1080 @ 30 FPS", "short_label": "1920x1080"},
            {"width": 1280, "height": 720, "fps": 30.0, "media_type": "RGB", "label": "1280x720 @ 30 FPS", "short_label": "1280x720"},
            {"width": 640, "height": 480, "fps": 30.0, "media_type": "RGB", "label": "640x480 @ 30 FPS", "short_label": "640x480"},
        ]
        self.available_formats = standard
        return standard

    def get_resolution_labels(self, cam_name: Optional[str] = None) -> List[str]:
        """
        Returns list of human-readable resolution labels with their theoretical FPS.
        e.g. ['1920x1080 (max 60 FPS)', '1280x720 (max 60 FPS)', '640x480 (max 90 FPS)']
        """
        formats = self.scan_resolutions_and_fps(cam_name)
        # Deduplicate by width x height, picking the highest FPS for each resolution
        res_map: Dict[Tuple[int, int], float] = {}
        for f in formats:
            key = (f["width"], f["height"])
            if key not in res_map or f["fps"] > res_map[key]:
                res_map[key] = f["fps"]

        # Ensure standard downscaled resolutions (e.g. 1920x1080, 1280x720, 640x480) are available
        # even if the device driver only advertises a single native pin mode (like OBS Virtual Camera)
        if res_map:
            max_w = max(k[0] for k in res_map)
            max_h = max(k[1] for k in res_map)
            max_fps = max(res_map.values()) if res_map else 30.0
            common_standards = [
                (3840, 2160),
                (2560, 1440),
                (1920, 1080),
                (1280, 720),
                (640, 480),
            ]
            for std_w, std_h in common_standards:
                if std_w <= max_w and std_h <= max_h and (std_w, std_h) not in res_map:
                    res_map[(std_w, std_h)] = max_fps

        # Sort descending by area
        sorted_res = sorted(res_map.items(), key=lambda item: item[0][0] * item[0][1], reverse=True)
        labels = []
        for (w, h), fps in sorted_res:
            if fps > 0:
                labels.append(f"{w}x{h} (max {fps:.0f} FPS)")
            else:
                labels.append(f"{w}x{h}")
        return labels

    def init(self, cam_name: str) -> Tuple[bool, Any]:
        """Initialize capture and get current format and theoretical FPS."""
        devices = self.list_devices()
        try:
            self.cam_index = [idx for idx, name in enumerate(devices) if name == cam_name][0]
            self.cam_name = cam_name
        except IndexError:
            logger.error(f"Webcam '{cam_name}' not found.")
            return False, "Device not found"

        # Release existing capture
        if self.cap is not None:
            self.cap.release()

        # Open capture using DirectShow backend, fallback to MSMF / ANY if needed.
        self.cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
        if not self.cap or not self.cap.isOpened():
            logger.warning(f"CAP_DSHOW failed for {cam_name} (index {self.cam_index}), trying CAP_MSMF...")
            self.cap = cv2.VideoCapture(self.cam_index, cv2.CAP_MSMF)
        if not self.cap or not self.cap.isOpened():
            logger.warning(f"CAP_MSMF failed for {cam_name}, trying default CAP_ANY...")
            self.cap = cv2.VideoCapture(self.cam_index, cv2.CAP_ANY)

        if not self.cap or not self.cap.isOpened():
            logger.error(f"Could not open camera {cam_name} (index {self.cam_index})")
            return False, "Capture init failed"

        # Query active parameters
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_prop = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.theoretical_fps = fps_prop if fps_prop > 0 else 30.0

        # Scan formats
        self.scan_resolutions_and_fps(cam_name)

        logger.info(f"Webcam '{cam_name}' initialized at {self.width}x{self.height} @ ~{self.theoretical_fps:.1f} FPS")
        return True, (self.width, self.height, self.theoretical_fps)

    def read(self) -> Tuple[bool, Optional[cv2.Mat]]:
        """Read a single frame."""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return True, frame
        return False, None

    def close(self) -> None:
        """Release capture handles."""
        if self.cap:
            self.cap.release()
            self.cap = None
        self.running = False

    def open_settings(self) -> None:
        """Request the backend to show the native camera settings dialog (DSHOW usually)."""
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_SETTINGS, 1)

    def set_resolution(self, width: int, height: int, target_fps: Optional[float] = None) -> Tuple[bool, Any]:
        """Set capture resolution and optionally request target framerate."""
        if self.cap and self.cap.isOpened():
            # If high fps is requested, prefer MJPG fourcc if supported
            if target_fps and target_fps >= 60:
                try:
                    self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                except Exception:
                    pass

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if target_fps:
                self.cap.set(cv2.CAP_PROP_FPS, target_fps)

            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps_prop = float(self.cap.get(cv2.CAP_PROP_FPS))
            if fps_prop > 0:
                self.theoretical_fps = fps_prop

            return True, (self.width, self.height, self.theoretical_fps)
        return False, None

