from __future__ import annotations
from typing import Any, Dict, Tuple, Optional
import threading
import time
import numpy as np
import cv2
import dearpygui.dearpygui as dpg
from loguru import logger

from core.window_base import WindowBase
from core.input_output_types import IOTypes
from config.display_scaling import display_scaling
from core.config_manager import config
from core.app_state import app_state
from .rtsp import Rtsp


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

    description: str = "Control RTSP network IP cameras. Supports credentials, continuous/trigger modes, and auto-reconnection."

    url_input_tag: str
    user_input_tag: str
    password_input_tag: str
    connect_btn_tag: str
    camstatus_tag: str
    caminfo_tag: str
    mode_tag: str
    trigger_btn_tag: str

    cam_running: bool
    cam_thread: Optional[threading.Thread]
    acquisition_mode: str
    trigger_requested: bool

    rtsp_url: str
    rtsp_user: str
    rtsp_password: str

    rtsp: Rtsp

    def __init__(
        self,
        label: str = "RTSP Camera",
        win_width: int = display_scaling.scale(320),
        win_height: int = display_scaling.scale(320),
        pos: Tuple[int, int] = (100, 100),
        uuid: Optional[str] = None,
        outputs: Optional[Dict[str, Any]] = None,
        visible: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the RTSP Camera window module.

        Args:
            label: Window title displayed in the UI.
            win_width: Initial window width in pixels.
            win_height: Initial window height in pixels.
            pos: Initial (x, y) window position on screen.
            uuid: Optional unique module identifier.
            outputs: Optional predefined output connections dictionary.
            visible: Whether the window is visible upon creation.
            **kwargs: Additional keyword arguments forwarded to WindowBase.
        """
        # Define persistent fields first so super().__init__ loads them from kwargs
        self._persistent_fields = ["label", "rtsp_url", "rtsp_user", "rtsp_password", "acquisition_mode"]

        # Default values if not present in kwargs (deserialization)
        self.rtsp_url = ""
        self.rtsp_user = ""
        self.rtsp_password = ""
        self.acquisition_mode = "continuous"

        super().__init__(
            label=label,
            pos=pos,
            win_width=win_width,
            win_height=win_height,
            uuid=uuid,
            outputs=outputs,
            visible=visible,
            **kwargs,
        )

        self.url_input_tag = f"rtsp_url_input_{self.UUID}"
        self.user_input_tag = f"rtsp_user_input_{self.UUID}"
        self.password_input_tag = f"rtsp_password_input_{self.UUID}"
        self.connect_btn_tag = f"rtsp_connect_{self.UUID}"
        self.camstatus_tag = f"rtsp_status_{self.UUID}"
        self.caminfo_tag = f"rtsp_info_{self.UUID}"
        self.mode_tag = f"rtsp_mode_{self.UUID}"
        self.trigger_btn_tag = f"rtsp_trigger_{self.UUID}"

        self.cam_running = False
        self.cam_thread = None
        self.trigger_requested = False

        self.rtsp = Rtsp()

        self.accepted_input_types = [IOTypes.CMD_DICT]
        self.outputs = {
            "Color Frame (BGR)": IOTypes.FRAME,
        }
        self.connections = {k: [] for k in self.outputs}

        self._build_ui()
        self.update_permission()

    def _build_ui(self) -> None:
        """
        Construct DearPyGui widgets for stream URL, credentials, connection, and acquisition mode.
        """
        with dpg.window(
            label=self.label,
            width=self.win_width,
            height=self.win_height,
            pos=self.pos,
            tag=self.winID,
            show=self.visible,
        ):
            dpg.add_text("RTSP Connection Settings", color=(150, 200, 150))

            dpg.add_input_text(
                label="Stream URL",
                default_value=self.rtsp_url,
                tag=self.url_input_tag,
                callback=self._on_url_changed,
                hint="192.168.1.100:554/stream1",
            )

            dpg.add_input_text(
                label="Username",
                default_value=self.rtsp_user,
                tag=self.user_input_tag,
                callback=self._on_user_changed,
                hint="admin",
            )

            dpg.add_input_text(
                label="Password",
                default_value=self.rtsp_password,
                tag=self.password_input_tag,
                callback=self._on_password_changed,
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
                    width=display_scaling.scale(120),
                )
                dpg.add_button(
                    label="Trigger Frame",
                    tag=self.trigger_btn_tag,
                    callback=self._on_trigger,
                    enabled=(self.acquisition_mode == "trigger"),
                    width=-1,
                )

        self.autosize_window()

    def update_permission(self) -> None:
        """
        Adjust module permissions and UI visibility based on the application mode.
        """
        is_user = app_state.mode == "user"

        if dpg.does_item_exist(self.winID):
            dpg.configure_item(self.winID, no_close=is_user)

    def _on_url_changed(self, sender: Any, app_data: str, *args: Any, **kwargs: Any) -> None:
        """
        Update stored RTSP stream URL from user input.
        """
        self.rtsp_url = app_data.strip()

    def _on_user_changed(self, sender: Any, app_data: str, *args: Any, **kwargs: Any) -> None:
        """
        Update stored RTSP username from user input.
        """
        self.rtsp_user = app_data

    def _on_password_changed(self, sender: Any, app_data: str, *args: Any, **kwargs: Any) -> None:
        """
        Update stored RTSP password from user input.
        """
        self.rtsp_password = app_data

    def _toggle_password_visibility(self, sender: Any, app_data: bool, *args: Any, **kwargs: Any) -> None:
        """
        Toggle password input field masking on or off.
        """
        if dpg.does_item_exist(self.password_input_tag):
            dpg.configure_item(self.password_input_tag, password=not app_data)

    def _on_mode_changed(self, sender: Any, app_data: str, *args: Any, **kwargs: Any) -> None:
        """
        Switch between Continuous streaming and software-Triggered frame acquisition.
        """
        if app_data == "Continuous":
            self.acquisition_mode = "continuous"
            if dpg.does_item_exist(self.trigger_btn_tag):
                dpg.configure_item(self.trigger_btn_tag, enabled=False)
        else:
            self.acquisition_mode = "trigger"
            if dpg.does_item_exist(self.trigger_btn_tag):
                dpg.configure_item(self.trigger_btn_tag, enabled=True)

    def _on_trigger(self, *args: Any, **kwargs: Any) -> None:
        """
        Request emission of the next grabbed frame in Trigger mode.
        """
        self.trigger_requested = True

    def _toggle_connection(self, *args: Any, **kwargs: Any) -> None:
        """
        Toggle camera connection state (Connect/Disconnect button callback).
        """
        if self.cam_running:
            self.disconnect_camera()
        else:
            self.connect_camera()

    def connect_camera(self) -> None:
        """
        Start the background acquisition thread and attempt to open the RTSP stream.
        """
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

            # Connect (this can block for a few seconds if camera is unreachable)
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
                ret_read, frame = self.rtsp.read()
                if ret_read and frame is not None:
                    reconnect_attempts = 0
                    if self.acquisition_mode == "continuous":
                        self.trigger_cb(frame)
                    elif self.acquisition_mode == "trigger" and self.trigger_requested:
                        self.trigger_requested = False
                        self.trigger_cb(frame)
                    # Small sleep to prevent pegging CPU when reading very fast (e.g. if local file / loopback)
                    time.sleep(0.005)
                else:
                    if not self.cam_running:
                        break
                    logger.warning("RTSP stream read failed. Attempting reconnect...")
                    if dpg.does_item_exist(self.camstatus_tag):
                        dpg.set_value(self.camstatus_tag, "Status: Reconnecting...")
                    self.rtsp.close()

                    # Sleep before reconnecting to prevent infinite fast-looping
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
        """
        Stop frame acquisition and release the RTSP stream connection.
        """
        self.cam_running = False
        # release VideoCapture to interrupt any blocked reads
        self.rtsp.close()
        if self.cam_thread and self.cam_thread.is_alive() and threading.current_thread() != self.cam_thread:
            self.cam_thread.join(timeout=1.0)

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
        angle = config.get("Cameras", {}).get("RTSP", {}).get("rotation", 0)
        if angle == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def trigger_cb(self, frame: cv2.Mat | np.ndarray) -> None:
        """
        Emit a frame to all connected downstream modules.

        Args:
            frame: Raw or rotated image frame from the RTSP stream.
        """
        frame = self._apply_rotation(frame)
        frame_tuple = (frame, f"RTSPCam_{self.UUID[:8]}")
        output_key = "Color Frame (BGR)"

        connected_modules = list(self.connections.get(output_key, []))
        for module in connected_modules:
            try:
                if hasattr(module, "input_cb"):
                    module.input_cb(data=frame_tuple, data_type=IOTypes.FRAME)
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

        if cmd and isinstance(cmd, dict):
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

    def is_outputs_ready(self) -> bool:
        """
        Check if downstream modules connected to this window are ready.

        Returns:
            bool: Always True for the camera stream.
        """
        return True

    def on_close(self) -> None:
        """
        Cleanup logic executed before window closure.
        """
        self.disconnect_camera()
        super().on_close()

    def close(self) -> None:
        """
        Stop acquisition, release stream resources, and unregister module.
        """
        self.disconnect_camera()
        super().close()


EXPORTED_CLASS = Rtsp_win
EXPORTED_NAME = "RTSP Camera"
