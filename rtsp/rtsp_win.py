from __future__ import annotations

import threading
import time
from typing import Any, Optional

import cv2
import dearpygui.dearpygui as dpg
import numpy as np
from loguru import logger

from config.display_scaling import display_scaling
from core.app_state import app_state
from core.config_manager import config
from core.input_output_types import IOTypes
from core.window_base import WindowBase
from .rtsp import Rtsp

_ROTATION_MAP = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


class Rtsp_win(WindowBase):
    """
    Window module for controlling RTSP Network/IP cameras.

    Features:
        - Connection to standard RTSP video streams with authentication (user/password).
        - Continuous live streaming and software-triggered capture modes.
        - Background worker thread with automatic reconnection on stream dropout.
        - Optional frame rotation via global configuration (`Cameras.RTSP.rotation`).
        - Remote automation control via dictionary commands (`IOTypes.CMD_DICT`).
    """

    label: str = "RTSP Camera"
    win_width: int = display_scaling.scale(320)
    win_height: int = display_scaling.scale(320)
    description: str = (
        "Control RTSP network IP cameras. Supports credentials, continuous/trigger modes, and auto-reconnection."
    )

    accepted_input_types: list[str] = [IOTypes.CMD_DICT]
    outputs: dict[str, str] = {
        "Color Frame (BGR)": IOTypes.FRAME,
    }
    persistent_fields: list[str] = [
        "label",
        "rtsp_url",
        "rtsp_user",
        "rtsp_password",
        "acquisition_mode",
    ]

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the RTSP Camera window module.

        Args:
            **kwargs: Additional keyword arguments forwarded to WindowBase.
        """
        super().__init__(**kwargs)

        uid = self.UUID
        self.url_input_tag: str = f"rtsp_url_input_{uid}"
        self.user_input_tag: str = f"rtsp_user_input_{uid}"
        self.password_input_tag: str = f"rtsp_password_input_{uid}"
        self.connect_btn_tag: str = f"rtsp_connect_{uid}"
        self.camstatus_tag: str = f"rtsp_status_{uid}"
        self.caminfo_tag: str = f"rtsp_info_{uid}"
        self.mode_tag: str = f"rtsp_mode_{uid}"
        self.trigger_btn_tag: str = f"rtsp_trigger_{uid}"

        self.rtsp_url: str = str(getattr(self, "rtsp_url", None) or "")
        self.rtsp_user: str = str(getattr(self, "rtsp_user", None) or "")
        self.rtsp_password: str = str(getattr(self, "rtsp_password", None) or "")
        self.acquisition_mode: str = str(getattr(self, "acquisition_mode", None) or "continuous")

        self.cam_running: bool = False
        self.cam_thread: Optional[threading.Thread] = None
        self.trigger_requested: bool = False
        self.rtsp: Rtsp = Rtsp()

        angle = config.get("Cameras", {}).get("RTSP", {}).get("rotation", 0)
        self._rotation_code: Optional[int] = _ROTATION_MAP.get(angle)
        self._source_name: str = f"RTSPCam_{self.UUID[:8]}"

        self._build_ui()
        self.autosize_window()
        self.update_permission()

    def _build_ui(self) -> None:
        """Construct DearPyGui widgets for stream URL, credentials, connection, and acquisition mode."""
        scale = display_scaling.scale

        with self.window():
            dpg.add_text("RTSP Connection Settings", color=(150, 200, 150))

            dpg.add_input_text(
                label="Stream URL",
                default_value=self.rtsp_url,
                tag=self.url_input_tag,
                callback=lambda s, a: setattr(self, "rtsp_url", a.strip()),
                hint="192.168.1.100:554/stream1",
            )

            dpg.add_input_text(
                label="Username",
                default_value=self.rtsp_user,
                tag=self.user_input_tag,
                callback=lambda s, a: setattr(self, "rtsp_user", a),
                hint="admin",
            )

            dpg.add_input_text(
                label="Password",
                default_value=self.rtsp_password,
                tag=self.password_input_tag,
                callback=lambda s, a: setattr(self, "rtsp_password", a),
                password=True,
                hint="password",
            )

            with dpg.group(horizontal=True):
                dpg.add_checkbox(
                    label="Show Password",
                    callback=self._toggle_password_visibility,
                )

            dpg.add_separator()

            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Connect",
                    tag=self.connect_btn_tag,
                    callback=self._toggle_connection,
                    width=-1,
                )

            dpg.add_text("Status: Disconnected", tag=self.camstatus_tag)
            dpg.add_text("Info: NA", tag=self.caminfo_tag)

            dpg.add_separator()

            dpg.add_text("Acquisition", color=(180, 180, 100))
            with dpg.group(horizontal=True):
                dpg.add_text("Mode:")
                dpg.add_combo(
                    items=["Continuous", "Trigger"],
                    default_value="Continuous" if self.acquisition_mode == "continuous" else "Trigger",
                    tag=self.mode_tag,
                    callback=self._on_mode_changed,
                    width=scale(120),
                )
                dpg.add_button(
                    label="Trigger Frame",
                    tag=self.trigger_btn_tag,
                    callback=self._on_trigger,
                    enabled=(self.acquisition_mode == "trigger"),
                    width=-1,
                )

    def update_permission(self) -> None:
        """Adjust module permissions and UI visibility based on the application mode."""
        is_user = app_state.mode == "user"
        if dpg.does_item_exist(self.winID):
            dpg.configure_item(self.winID, no_close=is_user)

    def _toggle_password_visibility(self, sender: Any = None, app_data: bool = False, *args: Any, **kwargs: Any) -> None:
        """Toggle password input field masking on or off."""
        if dpg.does_item_exist(self.password_input_tag):
            dpg.configure_item(self.password_input_tag, password=not app_data)

    def _on_mode_changed(self, sender: Any = None, app_data: str = "Continuous", *args: Any, **kwargs: Any) -> None:
        """Switch between Continuous streaming and software-Triggered frame acquisition."""
        if app_data == "Continuous":
            self.acquisition_mode = "continuous"
            if dpg.does_item_exist(self.trigger_btn_tag):
                dpg.configure_item(self.trigger_btn_tag, enabled=False)
        else:
            self.acquisition_mode = "trigger"
            if dpg.does_item_exist(self.trigger_btn_tag):
                dpg.configure_item(self.trigger_btn_tag, enabled=True)

    def _on_trigger(self, *args: Any, **kwargs: Any) -> None:
        """Request emission of the next grabbed frame in Trigger mode."""
        self.trigger_requested = True

    def _toggle_connection(self, *args: Any, **kwargs: Any) -> None:
        """Toggle camera connection state (Connect/Disconnect button callback)."""
        if self.cam_running:
            self.disconnect_camera()
        else:
            self.connect_camera()

    def connect_camera(self) -> None:
        """Start the background acquisition thread and attempt to open the RTSP stream."""
        if self.cam_thread and self.cam_thread.is_alive():
            self.cam_running = False
            self.cam_thread.join(timeout=1.0)

        self.cam_running = True
        if dpg.does_item_exist(self.connect_btn_tag):
            dpg.configure_item(self.connect_btn_tag, label="Disconnect")
        if dpg.does_item_exist(self.camstatus_tag):
            dpg.set_value(self.camstatus_tag, "Status: Connecting...")

        def task() -> None:
            url = self.rtsp_url
            user = self.rtsp_user
            password = self.rtsp_password

            ret, info = self.rtsp.init(url, user, password)

            if not ret:
                logger.error(f"RTSP connection failed: {info}")
                if self.cam_running:
                    if dpg.does_item_exist(self.camstatus_tag):
                        dpg.set_value(self.camstatus_tag, f"Status: Error ({info})")
                    if dpg.does_item_exist(self.connect_btn_tag):
                        dpg.configure_item(self.connect_btn_tag, label="Connect")
                    self.cam_running = False
                return

            if self.cam_running:
                if dpg.does_item_exist(self.camstatus_tag):
                    dpg.set_value(self.camstatus_tag, "Status: Connected")
                if dpg.does_item_exist(self.caminfo_tag) and isinstance(info, tuple):
                    w, h = info
                    dpg.set_value(self.caminfo_tag, f"Format: {w}x{h}")

            reconnect_attempts = 0
            while self.cam_running:
                if self.acquisition_mode == "trigger" and not self.trigger_requested:
                    ok_grab = self.rtsp.grab()
                    if ok_grab:
                        reconnect_attempts = 0
                        time.sleep(0.005)
                        continue
                    ret_read, frame = False, None
                else:
                    ret_read, frame = self.rtsp.read()

                if ret_read and frame is not None:
                    reconnect_attempts = 0
                    if self.acquisition_mode == "continuous":
                        self.trigger_cb(frame)
                    elif self.acquisition_mode == "trigger" and self.trigger_requested:
                        self.trigger_requested = False
                        self.trigger_cb(frame)
                    time.sleep(0.005)
                else:
                    if not self.cam_running:
                        break
                    logger.warning("RTSP stream read failed. Attempting reconnect...")
                    if dpg.does_item_exist(self.camstatus_tag):
                        dpg.set_value(self.camstatus_tag, "Status: Reconnecting...")
                    self.rtsp.close()

                    time.sleep(1.0)
                    if not self.cam_running:
                        break

                    ok, conn_info = self.rtsp.init(url, user, password)
                    if ok:
                        logger.info("RTSP stream reconnected successfully.")
                        if dpg.does_item_exist(self.camstatus_tag):
                            dpg.set_value(self.camstatus_tag, "Status: Connected")
                        if dpg.does_item_exist(self.caminfo_tag) and isinstance(conn_info, tuple):
                            w, h = conn_info
                            dpg.set_value(self.caminfo_tag, f"Format: {w}x{h}")
                    else:
                        reconnect_attempts += 1
                        if dpg.does_item_exist(self.camstatus_tag):
                            dpg.set_value(self.camstatus_tag, f"Status: Reconnecting... ({reconnect_attempts})")
                        time.sleep(2.0)

            self.rtsp.close()

        self.cam_thread = threading.Thread(target=task, daemon=True)
        self.cam_thread.start()

    def disconnect_camera(self) -> None:
        """Stop frame acquisition and release the RTSP stream connection."""
        self.cam_running = False
        if self.cam_thread and self.cam_thread.is_alive() and threading.current_thread() != self.cam_thread:
            self.cam_thread.join(timeout=1.0)
        self.rtsp.close()

        if dpg.does_item_exist(self.connect_btn_tag):
            dpg.configure_item(self.connect_btn_tag, label="Connect")
        if dpg.does_item_exist(self.camstatus_tag):
            dpg.set_value(self.camstatus_tag, "Status: Disconnected")
        if dpg.does_item_exist(self.caminfo_tag):
            dpg.set_value(self.caminfo_tag, "Info: NA")

    def _apply_rotation(self, frame: np.ndarray) -> np.ndarray:
        """
        Rotate frame based on camera configuration (Cameras.RTSP.rotation).

        Args:
            frame: Input image array.

        Returns:
            np.ndarray: Rotated image array.
        """
        if self._rotation_code is not None:
            return cv2.rotate(frame, self._rotation_code)
        return frame

    def trigger_cb(self, frame: cv2.Mat | np.ndarray) -> None:
        """
        Emit a frame to all connected downstream modules.

        Args:
            frame: Raw or rotated image frame from the RTSP stream.
        """
        frame = self._apply_rotation(frame)
        data = (frame, self._source_name)
        output_key = "Color Frame (BGR)"

        connected_modules = list(self.connections.get(output_key, []))
        for module in connected_modules:
            try:
                if hasattr(module, "input_cb"):
                    module.input_cb(data=data, data_type=IOTypes.FRAME)
            except Exception as e:
                logger.error(f"RTSP Camera emit error: {e}")

    def input_cb(self, data: Any = None, *args: Any, **kwargs: Any) -> None:
        """
        Handle incoming command dictionary from connected modules or automation scripts.

        Supported command keys:
            - connect (bool): Connect to the RTSP camera stream.
            - disconnect (bool): Disconnect from the RTSP camera stream.
            - trigger (bool): Request acquisition of a single frame.
            - trigger_cam (bool): Request acquisition of a single frame.
            - mode (str): Acquisition mode ("Continuous" or "Trigger").
            - url (str): Set RTSP stream URL.
            - user (str): Set RTSP username.
            - password (str): Set RTSP password.
        """
        if data is None and "data" in kwargs:
            data = kwargs["data"]
        elif data is None and args:
            data = args[0]

        if isinstance(data, dict):
            cmd = data
        elif kwargs.get("data_type") == IOTypes.CMD_DICT:
            cmd = kwargs.get("data")
        else:
            cmd = None

        if not cmd or not isinstance(cmd, dict):
            return

        if "url" in cmd:
            self.rtsp_url = str(cmd["url"]).strip()
            if dpg.does_item_exist(self.url_input_tag):
                dpg.set_value(self.url_input_tag, self.rtsp_url)

        if "user" in cmd:
            self.rtsp_user = str(cmd["user"])
            if dpg.does_item_exist(self.user_input_tag):
                dpg.set_value(self.user_input_tag, self.rtsp_user)

        if "password" in cmd:
            self.rtsp_password = str(cmd["password"])
            if dpg.does_item_exist(self.password_input_tag):
                dpg.set_value(self.password_input_tag, self.rtsp_password)

        if "connect" in cmd and cmd["connect"]:
            if not self.cam_running:
                self.connect_camera()

        if "disconnect" in cmd and cmd["disconnect"]:
            if self.cam_running:
                self.disconnect_camera()

        if "mode" in cmd:
            mode_str = str(cmd["mode"]).capitalize()
            if mode_str in ("Continuous", "Trigger"):
                self._on_mode_changed(None, mode_str)
                if dpg.does_item_exist(self.mode_tag):
                    dpg.set_value(self.mode_tag, mode_str)

        if cmd.get("trigger") or cmd.get("trigger_cam"):
            self._on_trigger()

    def is_ready(self) -> bool:
        """
        Check if the RTSP camera module is ready to accept commands or frames.

        Returns:
            bool: Always True.
        """
        return True

    def on_close(self) -> None:
        """Cleanup logic executed before window closure."""
        self.disconnect_camera()
        super().on_close()

    def close(self) -> None:
        """Stop acquisition, release stream resources, and unregister module."""
        self.on_close()
        super().close()


EXPORTED_CLASS = Rtsp_win
EXPORTED_NAME = "RTSP Camera"
