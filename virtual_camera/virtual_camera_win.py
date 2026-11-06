from __future__ import annotations

from typing import Any, Optional

import dearpygui.dearpygui as dpg
import numpy as np
from loguru import logger

from config.display_scaling import display_scaling
from core.app_state import app_state
from core.input_output_types import IOTypes
from core.window_base import WindowBase
from .virtual_camera import VirtualCameraEngine


class Virtual_Camera_win(WindowBase):
    """
    Window module for streaming image frames to an OS-level virtual camera device
    (e.g., OBS Virtual Camera, Unity Capture, etc.).

    Features:
        - Multi-backend streaming (OBS, Unity Capture, v4l2loopback, Auto).
        - Color/depth auto-conversion (Grayscale, BGR, BGRA, high bit-depth normalization).
        - Dynamic resolution auto-fitting to incoming frame size.
        - Transparent downstream frame passthrough for pipeline chaining.
        - Real-time telemetry monitoring (device name, sent frames, live FPS).
        - Remote automation control via dictionary commands (`IOTypes.CMD_DICT`).
    """

    label: str = "Virtual Camera"
    win_width: int = display_scaling.scale(360)
    win_height: int = display_scaling.scale(420)
    description: str = (
        "Stream image frames to an OS-level virtual camera device (OBS Virtual Camera, Unity Capture, etc.)."
    )

    accepted_input_types: list[str] = [IOTypes.FRAME, IOTypes.CMD_DICT]
    outputs: dict[str, str] = {
        "Color Frame (BGR)": IOTypes.FRAME,
        "Status": IOTypes.STATUS_DICT,
    }
    persistent_fields: list[str] = [
        "label",
        "target_width",
        "target_height",
        "target_fps",
        "backend",
        "auto_fit_res",
    ]

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the Virtual Camera window module.

        Args:
            **kwargs: Additional keyword arguments forwarded to WindowBase.
        """
        super().__init__(**kwargs)

        uid = self.UUID
        self.start_btn_tag: str = f"vcam_start_btn_{uid}"
        self.backend_combo_tag: str = f"vcam_backend_{uid}"
        self.width_input_tag: str = f"vcam_width_{uid}"
        self.height_input_tag: str = f"vcam_height_{uid}"
        self.fps_input_tag: str = f"vcam_fps_{uid}"
        self.autofit_checkbox_tag: str = f"vcam_autofit_{uid}"
        self.status_text_tag: str = f"vcam_status_{uid}"
        self.device_text_tag: str = f"vcam_device_{uid}"
        self.stats_text_tag: str = f"vcam_stats_{uid}"
        self.error_text_tag: str = f"vcam_error_{uid}"

        self.target_width: int = int(getattr(self, "target_width", None) or 1280)
        self.target_height: int = int(getattr(self, "target_height", None) or 720)
        self.target_fps: int = int(getattr(self, "target_fps", None) or 30)

        raw_backend = getattr(self, "backend", None)
        backend_str = str(raw_backend if raw_backend is not None else "obs").lower().strip()
        if backend_str in ("unityvideo", "unity", "unity_capture"):
            backend_str = "unitycapture"
        elif backend_str in ("obs_virtualcam", "obsvirtualcam"):
            backend_str = "obs"
        self.backend: str = backend_str

        raw_autofit = getattr(self, "auto_fit_res", None)
        self.auto_fit_res: bool = True if raw_autofit is None else bool(raw_autofit)

        self.engine: VirtualCameraEngine = VirtualCameraEngine(
            width=self.target_width,
            height=self.target_height,
            fps=self.target_fps,
            backend=self.backend,
        )

        self._build_ui()
        self.update_permission()

    def _build_ui(self) -> None:
        """Construct the DearPyGui interface."""
        scale = display_scaling.scale

        with self.window():
            # Start / Stop Control
            dpg.add_button(
                tag=self.start_btn_tag,
                label="Start Virtual Camera",
                callback=self._on_toggle_camera,
                width=-1,
                height=scale(32),
            )

            dpg.add_separator()

            # Library Check Banner
            if not self.engine.is_library_available():
                dpg.add_text(
                    "WARNING: 'pyvirtualcam' is not installed!\nRun 'pip install pyvirtualcam' in terminal.",
                    color=(255, 100, 100),
                )
                dpg.add_separator()

            # Settings
            with dpg.collapsing_header(label="Settings", default_open=True):
                backends = self.engine.get_supported_backends()
                dpg.add_combo(
                    tag=self.backend_combo_tag,
                    label="Backend",
                    items=backends,
                    default_value=self.backend,
                    width=scale(180),
                    callback=self._on_backend_change,
                )

                dpg.add_checkbox(
                    tag=self.autofit_checkbox_tag,
                    label="Auto-fit to incoming frame size",
                    default_value=self.auto_fit_res,
                    callback=lambda s, v: setattr(self, "auto_fit_res", v),
                )

                dpg.add_input_int(
                    tag=self.width_input_tag,
                    label="Width",
                    default_value=self.target_width,
                    min_value=160,
                    max_value=3840,
                    width=scale(160),
                    callback=lambda s, v: setattr(self, "target_width", max(160, v)),
                )

                dpg.add_input_int(
                    tag=self.height_input_tag,
                    label="Height",
                    default_value=self.target_height,
                    min_value=120,
                    max_value=2160,
                    width=scale(160),
                    callback=lambda s, v: setattr(self, "target_height", max(120, v)),
                )

                dpg.add_input_int(
                    tag=self.fps_input_tag,
                    label="Target FPS",
                    default_value=self.target_fps,
                    min_value=1,
                    max_value=120,
                    width=scale(160),
                    callback=lambda s, v: setattr(self, "target_fps", max(1, v)),
                )

            dpg.add_separator()

            # Status & Telemetry
            with dpg.collapsing_header(label="Status & Telemetry", default_open=True):
                dpg.add_text(tag=self.status_text_tag, default_value="Status: Stopped", color=(180, 180, 180))
                dpg.add_text(tag=self.device_text_tag, default_value="Device: None", color=(180, 180, 180))
                dpg.add_text(tag=self.stats_text_tag, default_value="Frames: 0 | FPS: 0.0", color=(180, 180, 180))
                dpg.add_text(tag=self.error_text_tag, default_value="", color=(255, 120, 120))

        self.autosize_window()

    def update_permission(self) -> None:
        """Adapt module UI based on application permission mode."""
        is_user_mode = getattr(app_state, "mode", "dev") == "user"

        if dpg.does_item_exist(self.winID):
            dpg.configure_item(self.winID, no_close=is_user_mode)
        if dpg.does_item_exist(self.backend_combo_tag):
            dpg.configure_item(self.backend_combo_tag, enabled=not is_user_mode)
        if dpg.does_item_exist(self.width_input_tag):
            dpg.configure_item(self.width_input_tag, readonly=is_user_mode)
        if dpg.does_item_exist(self.height_input_tag):
            dpg.configure_item(self.height_input_tag, readonly=is_user_mode)
        if dpg.does_item_exist(self.fps_input_tag):
            dpg.configure_item(self.fps_input_tag, readonly=is_user_mode)

    def _on_backend_change(
        self,
        sender: Any = None,
        app_data: Any = None,
        user_data: Any = None,
        *args: Any,
        value: str = "",
        **kwargs: Any,
    ) -> None:
        """
        Handle virtual camera driver backend selection update.

        Args:
            sender: UI sender widget.
            app_data: DPG app_data value.
            user_data: DPG user data.
            value: Backend identifier (optional kwarg).
        """
        val = value if value else (app_data if isinstance(app_data, str) else "")
        if not val:
            return
        self.backend = val
        if dpg.does_item_exist(self.backend_combo_tag) and dpg.get_value(self.backend_combo_tag) != val:
            dpg.set_value(self.backend_combo_tag, val)
        if self.engine.is_running:
            self._restart_camera()

    def _on_toggle_camera(self, *args: Any, **kwargs: Any) -> None:
        """Toggle the virtual camera on/off."""
        if self.engine.is_running:
            self._stop_camera()
        else:
            self._start_camera()

    def _start_camera(self) -> None:
        """Start the virtual camera engine."""
        success, msg = self.engine.start(
            width=self.target_width,
            height=self.target_height,
            fps=self.target_fps,
            backend=self.backend,
        )

        if success:
            if dpg.does_item_exist(self.start_btn_tag):
                dpg.configure_item(self.start_btn_tag, label="Stop Virtual Camera")
            if dpg.does_item_exist(self.status_text_tag):
                dpg.configure_item(self.status_text_tag, default_value="Status: Running", color=(100, 255, 100))
            if dpg.does_item_exist(self.device_text_tag):
                dpg.configure_item(self.device_text_tag, default_value=f"Device: {self.engine.device_name}", color=(200, 255, 200))
            if dpg.does_item_exist(self.error_text_tag):
                dpg.configure_item(self.error_text_tag, default_value="")
        else:
            if dpg.does_item_exist(self.status_text_tag):
                dpg.configure_item(self.status_text_tag, default_value="Status: Error", color=(255, 100, 100))
            if dpg.does_item_exist(self.error_text_tag):
                dpg.configure_item(self.error_text_tag, default_value=f"Error: {msg}")

        self._send_status()

    def _stop_camera(self) -> None:
        """Stop the virtual camera engine."""
        self.engine.stop()

        if dpg.does_item_exist(self.start_btn_tag):
            dpg.configure_item(self.start_btn_tag, label="Start Virtual Camera")
        if dpg.does_item_exist(self.status_text_tag):
            dpg.configure_item(self.status_text_tag, default_value="Status: Stopped", color=(180, 180, 180))
        if dpg.does_item_exist(self.device_text_tag):
            dpg.configure_item(self.device_text_tag, default_value="Device: None", color=(180, 180, 180))
        if dpg.does_item_exist(self.stats_text_tag):
            dpg.configure_item(self.stats_text_tag, default_value="Frames: 0 | FPS: 0.0")

        self._send_status()

    def _restart_camera(self) -> None:
        """Restart the virtual camera with updated parameters."""
        self._stop_camera()
        self._start_camera()

    def input_cb(self, data: Any = None, *args: Any, **kwargs: Any) -> None:
        """
        Process incoming frame from upstream module or command dictionary.

        Accepts:
            - Frame data: np.ndarray or tuple (np.ndarray, str) of type IOTypes.FRAME.
            - Command dictionary (IOTypes.CMD_DICT):
                - start (bool): Start virtual camera stream.
                - stop (bool): Stop virtual camera stream.
                - backend (str): Set backend ('obs', 'unitycapture', 'auto', etc.).
                - width (int): Set virtual camera target width.
                - height (int): Set virtual camera target height.
                - fps (int): Set target FPS.
                - auto_fit (bool): Toggle auto-fit resolution to incoming frame.
        """
        if data is None and "data" in kwargs:
            data = kwargs["data"]
        elif data is None and args:
            data = args[0]

        if data is None:
            return

        if isinstance(data, dict) or kwargs.get("data_type") == IOTypes.CMD_DICT:
            cmd = data if isinstance(data, dict) else kwargs.get("data")
            if isinstance(cmd, dict):
                if "start" in cmd:
                    if cmd["start"]:
                        self._start_camera()
                    else:
                        self._stop_camera()
                if "stop" in cmd and cmd["stop"]:
                    self._stop_camera()
                if "backend" in cmd:
                    self._on_backend_change(value=str(cmd["backend"]))
                if "width" in cmd:
                    w = max(160, int(cmd["width"]))
                    self.target_width = w
                    if dpg.does_item_exist(self.width_input_tag):
                        dpg.set_value(self.width_input_tag, w)
                if "height" in cmd:
                    h = max(120, int(cmd["height"]))
                    self.target_height = h
                    if dpg.does_item_exist(self.height_input_tag):
                        dpg.set_value(self.height_input_tag, h)
                if "fps" in cmd:
                    f = max(1, int(cmd["fps"]))
                    self.target_fps = f
                    if dpg.does_item_exist(self.fps_input_tag):
                        dpg.set_value(self.fps_input_tag, f)
                if "auto_fit" in cmd:
                    self.auto_fit_res = bool(cmd["auto_fit"])
                    if dpg.does_item_exist(self.autofit_checkbox_tag):
                        dpg.set_value(self.autofit_checkbox_tag, self.auto_fit_res)
            return

        frame = data[0] if isinstance(data, tuple) else data
        if not isinstance(frame, np.ndarray):
            return

        if self.auto_fit_res and not self.engine.is_running:
            h, w = frame.shape[:2]
            if w > 0 and h > 0 and (w != self.target_width or h != self.target_height):
                self.target_width = w
                self.target_height = h
                if dpg.does_item_exist(self.width_input_tag):
                    dpg.set_value(self.width_input_tag, w)
                if dpg.does_item_exist(self.height_input_tag):
                    dpg.set_value(self.height_input_tag, h)

        if self.engine.is_running:
            self.engine.send_frame(frame)

            if dpg.does_item_exist(self.stats_text_tag):
                dpg.set_value(
                    self.stats_text_tag,
                    f"Frames: {self.engine.frames_sent} | FPS: {self.engine.current_fps}",
                )

        self.trigger_cb(frame=data)

    def trigger_cb(self, frame: Any) -> None:
        """
        Send passthrough frame and status dictionary to connected downstream modules.

        Args:
            frame: Input frame tuple or numpy array.
        """
        bgr_modules = self.connections.get("Color Frame (BGR)")
        if bgr_modules:
            for module in list(bgr_modules):
                try:
                    if hasattr(module, "input_cb"):
                        module.input_cb(data=frame, data_type=IOTypes.FRAME)
                except Exception as e:
                    logger.error(f"[{self.label}] Passthrough error on 'Color Frame (BGR)': {e}")

        status_modules = self.connections.get("Status")
        if status_modules:
            status_data = {
                "is_running": self.engine.is_running,
                "device": self.engine.device_name,
                "fps": self.engine.current_fps,
                "frames_sent": self.engine.frames_sent,
            }
            for module in list(status_modules):
                try:
                    if hasattr(module, "input_cb"):
                        module.input_cb(data=status_data, data_type=IOTypes.STATUS_DICT)
                except Exception as e:
                    logger.error(f"[{self.label}] Passthrough error on 'Status': {e}")

    def _send_status(self) -> None:
        """Broadcast current status dictionary to connected status listeners."""
        status_data = {
            "is_running": self.engine.is_running,
            "device": self.engine.device_name,
            "fps": self.engine.current_fps,
            "frames_sent": self.engine.frames_sent,
        }
        connected_modules = list(self.connections.get("Status", []))
        for module in connected_modules:
            try:
                if hasattr(module, "input_cb"):
                    module.input_cb(data=status_data, data_type=IOTypes.STATUS_DICT)
            except Exception as e:
                logger.error(f"[{self.label}] Status broadcast error: {e}")

    def is_ready(self) -> bool:
        """
        Check if virtual camera module is ready.

        Returns:
            bool: Always True.
        """
        return True

    def on_close(self) -> None:
        """Clean up virtual camera on module close."""
        self.engine.stop()
        super().on_close()

    def close(self) -> None:
        """Stop virtual camera engine, release drivers, and close window."""
        self.on_close()
        super().close()


EXPORTED_CLASS = Virtual_Camera_win
EXPORTED_NAME = "Virtual Camera"
