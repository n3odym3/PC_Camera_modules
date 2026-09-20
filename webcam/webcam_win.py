from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
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
from .webcam import Webcam


class Webcam_win(WindowBase):
    """
    Window module for controlling generic USB Webcams via DirectShow and OpenCV.

    Features:
        - Auto-discovery of connected DirectShow video grabbers.
        - Hardware capability and resolution probing with theoretical maximum framerate display.
        - Live measured acquisition FPS monitoring alongside theoretical driver limits.
        - Native DirectShow hardware / UVC property dialog access.
        - Continuous live acquisition and software-triggered capture modes.
        - Configurable image orientation / rotation via `Cameras.Webcam.rotation`.
        - Remote automation control via dictionary commands (`IOTypes.CMD_DICT`).
    """

    description: str = "Control generic USB webcams. Supports resolution scanning, UVC controls, and live FPS monitoring."

    refresh_tag: str
    camlist_tag: str
    camstatus_tag: str
    caminfo_tag: str
    res_list_tag: str
    mode_tag: str
    trigger_btn_tag: str
    start_btn_tag: str
    stop_btn_tag: str

    camlist: List[str]
    cam_running: bool
    cam_thread: Optional[threading.Thread]

    acquisition_mode: str
    trigger_requested: bool
    _last_frame: Optional[np.ndarray]
    _last_frame_lock: threading.Lock

    webcam: Webcam

    def __init__(
        self,
        label: str = "Webcam",
        win_width: int = display_scaling.scale(320),
        win_height: int = display_scaling.scale(460),
        pos: Tuple[int, int] = (100, 100),
        uuid: Optional[str] = None,
        outputs: Optional[Dict[str, Any]] = None,
        visible: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the USB Webcam window module.

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

        self.refresh_tag = f"webcam_refresh_{self.UUID}"
        self.camlist_tag = f"webcam_camlist_{self.UUID}"
        self.camstatus_tag = f"webcam_status_{self.UUID}"
        self.caminfo_tag = f"webcam_info_{self.UUID}"
        self.res_list_tag = f"webcam_res_list_{self.UUID}"
        self.mode_tag = f"webcam_mode_{self.UUID}"
        self.trigger_btn_tag = f"webcam_trigger_{self.UUID}"
        self.start_btn_tag = f"webcam_start_{self.UUID}"
        self.stop_btn_tag = f"webcam_stop_{self.UUID}"

        self.camlist = []
        self.cam_running = False
        self.cam_thread = None

        self.acquisition_mode = "continuous"
        self.trigger_requested = False
        self._last_frame = None
        self._last_frame_lock = threading.Lock()

        self.webcam = Webcam()

        self._persistent_fields = ["label"]
        self.accepted_input_types = [IOTypes.CMD_DICT]
        self.outputs = {
            "Color Frame (BGR)": IOTypes.FRAME,
        }
        self.connections = {k: [] for k in self.outputs}

        self._build_ui()
        self.update_permission()

    def _build_ui(self) -> None:
        """
        Construct DearPyGui widgets for camera enumeration, streaming controls, and resolution selection.
        """
        with dpg.window(
            label=self.label,
            width=self.win_width,
            height=self.win_height,
            pos=self.pos,
            tag=self.winID,
            show=self.visible,
        ):
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Refresh", tag=self.refresh_tag, callback=self.refresh_cam
                )
                dpg.add_combo(
                    items=self.camlist,
                    tag=self.camlist_tag,
                    width=-1,
                    callback=self.select_cam,
                )

            dpg.add_text("Status: not initialized", tag=self.camstatus_tag)
            dpg.add_text("Info: NA", tag=self.caminfo_tag)

            dpg.add_separator()

            dpg.add_text("Acquisition", color=(180, 180, 100))
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Start",
                    tag=self.start_btn_tag,
                    callback=self.start_webcam_thread,
                    width=display_scaling.scale(65),
                    enabled=True,
                )
                dpg.add_button(
                    label="Stop",
                    tag=self.stop_btn_tag,
                    callback=self.stop_webcam_thread,
                    width=display_scaling.scale(65),
                    enabled=False,
                )
                dpg.add_text("Mode:")
                dpg.add_combo(
                    items=["Continuous", "Trigger"],
                    default_value="Continuous",
                    tag=self.mode_tag,
                    callback=self._on_mode_changed,
                    width=display_scaling.scale(100),
                )
                dpg.add_button(
                    label="Trigger",
                    tag=self.trigger_btn_tag,
                    callback=self._on_trigger,
                    enabled=False,
                    width=-1,
                )

            dpg.add_separator()

            dpg.add_text("UVC Controls", color=(150, 200, 150))
            dpg.add_button(
                label="Open Camera Settings",
                callback=self._on_open_settings,
                width=-1,
            )

            dpg.add_separator()
            dpg.add_text("Resolution & Capabilities", color=(150, 150, 200))
            dpg.add_combo(
                items=[],
                label="",
                tag=self.res_list_tag,
                width=-1,
                callback=self.change_resolution,
            )

        self.refresh_cam()
        self.autosize_window()

    def update_permission(self) -> None:
        """
        Adjust module permissions and UI visibility based on the application mode.
        """
        is_user = getattr(app_state, "mode", "dev") == "user"
        if dpg.does_item_exist(self.winID):
            dpg.configure_item(self.winID, no_close=is_user)

    def refresh_cam(self, *args: Any, **kwargs: Any) -> None:
        """
        Query available video capture devices and update the combo dropdown list.
        """
        self.camlist = self.webcam.list_devices()
        dpg.configure_item(self.camlist_tag, items=self.camlist)
        if self.camlist and dpg.does_item_exist(self.camlist_tag):
            if not dpg.get_value(self.camlist_tag):
                dpg.set_value(self.camlist_tag, self.camlist[0])

    def select_cam(self, sender: Any, cam_name: str, *args: Any, **kwargs: Any) -> None:
        """
        Initialize the chosen camera and dynamically scan available resolutions and theoretical FPS.

        Args:
            cam_name: Name of the video grabber device.
        """
        self.stop_webcam_thread()
        ret, info = self.webcam.init(cam_name)
        self.cam_running = ret

        if ret:
            # Query hardware-supported resolutions with theoretical frame rates
            labels = self.webcam.get_resolution_labels(cam_name)
            dpg.configure_item(self.res_list_tag, items=labels)

            w, h, th_fps = (
                (info[0], info[1], info[2])
                if isinstance(info, tuple) and len(info) >= 3
                else (self.webcam.width, self.webcam.height, self.webcam.theoretical_fps)
            )

            # Auto-select the preferred resolution (highest available in labels)
            active_label = labels[0] if labels else ""
            if active_label:
                dpg.set_value(self.res_list_tag, active_label)
                # Apply preferred resolution to OpenCV capture if different from initial default
                if not active_label.startswith(f"{w}x{h}"):
                    self.change_resolution(None, active_label)

            w, h = self.webcam.width, self.webcam.height
            th_fps = self.webcam.theoretical_fps

            if dpg.does_item_exist(self.camstatus_tag):
                dpg.set_value(self.camstatus_tag, "Status: Running")
            if dpg.does_item_exist(self.caminfo_tag):
                dpg.set_value(self.caminfo_tag, f"Format: {w}x{h} | Th. FPS: {th_fps:.1f} FPS")

            self.start_webcam_thread()
        else:
            if dpg.does_item_exist(self.camstatus_tag):
                dpg.set_value(self.camstatus_tag, f"Status: Error ({info})")

    def _on_mode_changed(self, sender: Any, app_data: Any, *args: Any, **kwargs: Any) -> None:
        """
        Switch between Continuous streaming and software Trigger mode.
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
        Request dispatch of the next acquired frame in Trigger mode.
        """
        self.trigger_requested = True

    def stop_webcam_thread(self, *args: Any, **kwargs: Any) -> None:
        """
        Stop the background frame acquisition worker thread.
        """
        self.cam_running = False
        if self.cam_thread and self.cam_thread.is_alive() and threading.current_thread() != self.cam_thread:
            self.cam_thread.join(timeout=1.0)
        self.cam_thread = None

        if dpg.does_item_exist(self.camstatus_tag):
            dpg.set_value(self.camstatus_tag, "Status: Stopped")
        if dpg.does_item_exist(self.start_btn_tag):
            dpg.configure_item(self.start_btn_tag, enabled=True)
        if dpg.does_item_exist(self.stop_btn_tag):
            dpg.configure_item(self.stop_btn_tag, enabled=False)

    def start_webcam_thread(self, *args: Any, **kwargs: Any) -> None:
        """
        Start the background frame grabber and FPS tracking worker thread.
        """
        if self.webcam.cap is None or not self.webcam.cap.isOpened():
            target = dpg.get_value(self.camlist_tag) if dpg.does_item_exist(self.camlist_tag) else ""
            if not target and self.camlist:
                target = self.camlist[0]
            if target:
                self.select_cam(None, target)
                return
            logger.warning("No webcam selected to start")
            return

        if self.cam_thread and self.cam_thread.is_alive() and threading.current_thread() != self.cam_thread:
            self.cam_running = False
            self.cam_thread.join(timeout=1.0)

        self.cam_running = True
        if dpg.does_item_exist(self.camstatus_tag):
            dpg.set_value(self.camstatus_tag, "Status: Running")
        if dpg.does_item_exist(self.start_btn_tag):
            dpg.configure_item(self.start_btn_tag, enabled=False)
        if dpg.does_item_exist(self.stop_btn_tag):
            dpg.configure_item(self.stop_btn_tag, enabled=True)

        def task() -> None:
            fps_count = 0
            fps_start = time.perf_counter()
            live_fps = 0.0

            while self.cam_running:
                ret, frame = self.webcam.read()
                if ret and frame is not None:
                    fps_count += 1
                    now = time.perf_counter()
                    if now - fps_start >= 1.0:
                        live_fps = fps_count / (now - fps_start)
                        fps_count = 0
                        fps_start = now
                        if dpg.does_item_exist(self.caminfo_tag):
                            th = self.webcam.theoretical_fps
                            w, h = self.webcam.width, self.webcam.height
                            dpg.set_value(self.caminfo_tag, f"Format: {w}x{h} | Th. {th:.0f} FPS | Live: {live_fps:.1f} FPS")

                    with self._last_frame_lock:
                        self._last_frame = frame

                    if self.acquisition_mode == "continuous":
                        self.trigger_cb(frame)
                    elif self.acquisition_mode == "trigger" and self.trigger_requested:
                        self.trigger_requested = False
                        self.trigger_cb(frame)
                else:
                    time.sleep(0.01)

        self.cam_thread = threading.Thread(target=task, daemon=True)
        self.cam_thread.start()

    def _on_open_settings(self, *args: Any, **kwargs: Any) -> None:
        """
        Open the native operating system DirectShow / UVC hardware settings dialog.
        """
        self.webcam.open_settings()

    def _apply_rotation(self, frame: np.ndarray) -> np.ndarray:
        """
        Rotate frame based on camera configuration (Cameras.Webcam.rotation).

        Args:
            frame: Input image array.

        Returns:
            np.ndarray: Rotated image array.
        """
        angle = config.get("Cameras", {}).get("Webcam", {}).get("rotation", 0)
        if angle == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def change_resolution(self, sender: Any, res_str: str, *args: Any, **kwargs: Any) -> None:
        """
        Change UVC stream resolution and requested framerate based on selection.

        Args:
            res_str: Formatted resolution label (e.g. "1920x1080 (max 60 FPS)").
        """
        if not res_str:
            return

        try:
            # Parse resolution prefix e.g. "1920x1080 (max 60 FPS)" -> "1920x1080"
            res_part = res_str.split()[0] if " " in res_str else res_str
            if "x" in res_part:
                w_str, h_str = res_part.split("x")
                w, h = int(w_str), int(h_str)

                # Extract requested FPS if mentioned
                target_fps = None
                if "max" in res_str:
                    try:
                        fps_part = res_str.split("max")[1].strip().split()[0]
                        target_fps = float(fps_part)
                    except Exception:
                        pass

                ret, info = self.webcam.set_resolution(w, h, target_fps=target_fps)
                if ret and isinstance(info, tuple):
                    w, h, th_fps = info
                    if dpg.does_item_exist(self.caminfo_tag):
                        dpg.set_value(self.caminfo_tag, f"Format: {w}x{h} | Th. FPS: {th_fps:.1f} FPS")
                    logger.info(f"Resolution changed to {w}x{h} @ {th_fps:.1f} FPS")
        except Exception as e:
            logger.error(f"Failed to change resolution: {e}")

    def trigger_cb(self, frame: cv2.Mat | np.ndarray) -> None:
        """
        Emit an acquired frame to all connected downstream modules.

        Args:
            frame: Image array from webcam.
        """
        frame = self._apply_rotation(frame)
        frame_tuple = (frame, f"Webcam_{self.webcam.cam_name}")
        output_key = "Color Frame (BGR)"

        connected_modules = list(self.connections.get(output_key, []))
        for module in connected_modules:
            try:
                if hasattr(module, "input_cb"):
                    module.input_cb(data=frame_tuple, data_type=IOTypes.FRAME)
            except Exception as e:
                logger.error(f"Webcam emit error: {e}")

    def input_cb(self, data: Any = None, *args: Any, **kwargs: Any) -> None:
        """
        Handle incoming command dictionary from connected modules or automation scripts.

        Supported command keys:
            - start (bool): Start continuous acquisition.
            - stop (bool): Stop continuous acquisition.
            - trigger (bool): Request acquisition of a single frame.
            - trigger_cam (bool): Request acquisition of a single frame.
            - mode (str): Acquisition mode ("Continuous" or "Trigger").
            - camera (str): Select and initialize camera by device name.
            - resolution (str): Switch resolution to a matching capability label.
            - open_settings (bool): Open the native UVC settings dialog.
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
            if "start" in cmd and cmd["start"]:
                self.start_webcam_thread()

            if "stop" in cmd and cmd["stop"]:
                self.stop_webcam_thread()

            if "mode" in cmd:
                mode_str = str(cmd["mode"]).capitalize()
                if mode_str in ("Continuous", "Trigger"):
                    self._on_mode_changed(None, mode_str)
                    if dpg.does_item_exist(self.mode_tag):
                        dpg.set_value(self.mode_tag, mode_str)

            if "camera" in cmd or "select_cam" in cmd:
                target_cam = str(cmd.get("camera") or cmd.get("select_cam"))
                self.select_cam(None, target_cam)

            if "resolution" in cmd:
                self.change_resolution(None, str(cmd["resolution"]))

            if "open_settings" in cmd and cmd["open_settings"]:
                self._on_open_settings()

            if cmd.get("trigger_cam") or cmd.get("trigger"):
                self._on_trigger()

    def is_ready(self) -> bool:
        """
        Check if the webcam module is ready to accept commands or frames.

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
        self.cam_running = False
        if self.cam_thread and self.cam_thread.is_alive() and threading.current_thread() != self.cam_thread:
            self.cam_thread.join(timeout=1.0)
        self.webcam.close()
        super().on_close()

    def close(self) -> None:
        """
        Stop acquisition, release camera resources, and unregister module.
        """
        self.cam_running = False
        if self.cam_thread and self.cam_thread.is_alive() and threading.current_thread() != self.cam_thread:
            self.cam_thread.join(timeout=1.0)
        self.webcam.close()
        super().close()


EXPORTED_CLASS = Webcam_win
EXPORTED_NAME = "Webcam"
