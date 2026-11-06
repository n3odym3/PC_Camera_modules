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
from .webcam import Webcam

_ROTATION_MAP = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


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

    label: str = "Webcam"
    win_width: int = display_scaling.scale(320)
    win_height: int = display_scaling.scale(460)
    description: str = (
        "Control generic USB webcams. Supports resolution scanning, UVC controls, and live FPS monitoring."
    )

    accepted_input_types: list[str] = [IOTypes.CMD_DICT]
    outputs: dict[str, str] = {
        "Color Frame (BGR)": IOTypes.FRAME,
    }
    persistent_fields: list[str] = ["label", "cam_name", "acquisition_mode"]

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the USB Webcam window module.

        Args:
            **kwargs: Additional keyword arguments forwarded to WindowBase.
        """
        super().__init__(**kwargs)

        uid = self.UUID
        self.refresh_tag: str = f"webcam_refresh_{uid}"
        self.camlist_tag: str = f"webcam_camlist_{uid}"
        self.camstatus_tag: str = f"webcam_status_{uid}"
        self.caminfo_tag: str = f"webcam_info_{uid}"
        self.res_list_tag: str = f"webcam_res_list_{uid}"
        self.mode_tag: str = f"webcam_mode_{uid}"
        self.trigger_btn_tag: str = f"webcam_trigger_{uid}"
        self.start_btn_tag: str = f"webcam_start_{uid}"
        self.stop_btn_tag: str = f"webcam_stop_{uid}"

        self.camlist: list[str] = []
        self.cam_running: bool = False
        self.cam_thread: Optional[threading.Thread] = None

        self.cam_name: str = str(getattr(self, "cam_name", None) or "")
        self.acquisition_mode: str = str(getattr(self, "acquisition_mode", None) or "continuous")
        self.trigger_requested: bool = False
        self._last_frame: Optional[np.ndarray] = None
        self._last_frame_lock: threading.Lock = threading.Lock()

        angle = config.get("Cameras", {}).get("Webcam", {}).get("rotation", 0)
        self._rotation_code: Optional[int] = _ROTATION_MAP.get(angle)
        self._source_name: str = f"Webcam_{self.cam_name}"

        self.webcam: Webcam = Webcam()

        self._build_ui()
        self.update_permission()

    def _build_ui(self) -> None:
        """
        Construct DearPyGui widgets for camera enumeration, streaming controls, and resolution selection.
        """
        scale = display_scaling.scale

        with self.window():
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
                    width=scale(65),
                    enabled=True,
                )
                dpg.add_button(
                    label="Stop",
                    tag=self.stop_btn_tag,
                    callback=self.stop_webcam_thread,
                    width=scale(65),
                    enabled=False,
                )
                dpg.add_text("Mode:")
                dpg.add_combo(
                    items=["Continuous", "Trigger"],
                    default_value=self.acquisition_mode.capitalize(),
                    tag=self.mode_tag,
                    callback=self._on_mode_changed,
                    width=scale(100),
                )
                dpg.add_button(
                    label="Trigger",
                    tag=self.trigger_btn_tag,
                    callback=self._on_trigger,
                    enabled=(self.acquisition_mode == "trigger"),
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
        if dpg.does_item_exist(self.camlist_tag):
            dpg.configure_item(self.camlist_tag, items=self.camlist)
            current_val = dpg.get_value(self.camlist_tag)
            if self.cam_name and self.cam_name in self.camlist:
                dpg.set_value(self.camlist_tag, self.cam_name)
            elif not current_val and self.camlist:
                dpg.set_value(self.camlist_tag, self.camlist[0])

    def select_cam(
        self,
        sender: Any = None,
        app_data: Any = None,
        user_data: Any = None,
        *args: Any,
        cam_name: str = "",
        **kwargs: Any,
    ) -> None:
        """
        Initialize the chosen camera and dynamically scan available resolutions and theoretical FPS.

        Args:
            sender: DPG sender tag.
            app_data: DPG app_data value.
            user_data: DPG user data.
            cam_name: Name of the video grabber device.
        """
        target = cam_name if cam_name else (app_data if isinstance(app_data, str) else "")
        if not target:
            return

        self.cam_name = target
        self._source_name = f"Webcam_{target}"
        self.stop_webcam_thread()
        ret, info = self.webcam.init(target)
        self.cam_running = ret
        self._source_name = f"Webcam_{self.webcam.cam_name}"

        if ret:
            labels = self.webcam.get_resolution_labels(target)
            if dpg.does_item_exist(self.res_list_tag):
                dpg.configure_item(self.res_list_tag, items=labels)

            w, h, th_fps = (
                (info[0], info[1], info[2])
                if isinstance(info, tuple) and len(info) >= 3
                else (self.webcam.width, self.webcam.height, self.webcam.theoretical_fps)
            )

            active_label = labels[0] if labels else ""
            if active_label and dpg.does_item_exist(self.res_list_tag):
                dpg.set_value(self.res_list_tag, active_label)
                if not active_label.startswith(f"{w}x{h}"):
                    self.change_resolution(res_str=active_label)

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

    def _on_mode_changed(
        self,
        sender: Any = None,
        app_data: Any = "Continuous",
        user_data: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Switch between Continuous streaming and software Trigger mode.
        """
        mode_val = str(app_data).strip().capitalize()
        if mode_val == "Continuous":
            self.acquisition_mode = "continuous"
            if dpg.does_item_exist(self.trigger_btn_tag):
                dpg.configure_item(self.trigger_btn_tag, enabled=False)
        elif mode_val == "Trigger":
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
            target = dpg.get_value(self.camlist_tag) if dpg.does_item_exist(self.camlist_tag) else self.cam_name
            if not target and self.camlist:
                target = self.camlist[0]
            if target:
                self.select_cam(cam_name=target)
                return
            logger.warning(f"[{self.label}] No webcam selected to start")
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
        if self._rotation_code is not None:
            return cv2.rotate(frame, self._rotation_code)
        return frame

    def change_resolution(
        self,
        sender: Any = None,
        app_data: Any = None,
        user_data: Any = None,
        *args: Any,
        res_str: str = "",
        **kwargs: Any,
    ) -> None:
        """
        Change UVC stream resolution and requested framerate based on selection.

        Args:
            sender: DPG sender tag.
            app_data: DPG app_data value.
            user_data: DPG user data.
            res_str: Formatted resolution label (e.g. "1920x1080 (max 60 FPS)").
        """
        target_str = res_str if res_str else (app_data if isinstance(app_data, str) else "")
        if not target_str:
            return

        try:
            res_part = target_str.split()[0] if " " in target_str else target_str
            if "x" in res_part:
                w_str, h_str = res_part.split("x")
                w, h = int(w_str), int(h_str)

                target_fps = None
                if "max" in target_str:
                    try:
                        fps_part = target_str.split("max")[1].strip().split()[0]
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
        data = (frame, self._source_name)
        output_key = "Color Frame (BGR)"

        connected_modules = list(self.connections.get(output_key, []))
        for module in connected_modules:
            try:
                if hasattr(module, "input_cb"):
                    module.input_cb(data=data, data_type=IOTypes.FRAME)
            except Exception as e:
                logger.error(f"[{self.label}] Webcam emit error: {e}")

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

        if not cmd or not isinstance(cmd, dict):
            return

        if cmd.get("start"):
            self.start_webcam_thread()

        if cmd.get("stop"):
            self.stop_webcam_thread()

        if "mode" in cmd:
            mode_str = str(cmd["mode"]).capitalize()
            if mode_str in ("Continuous", "Trigger"):
                self._on_mode_changed(app_data=mode_str)
                if dpg.does_item_exist(self.mode_tag):
                    dpg.set_value(self.mode_tag, mode_str)

        if "camera" in cmd or "select_cam" in cmd:
            target_cam = str(cmd.get("camera") or cmd.get("select_cam"))
            self.select_cam(cam_name=target_cam)

        if "resolution" in cmd:
            self.change_resolution(res_str=str(cmd["resolution"]))

        if cmd.get("open_settings"):
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

    def on_close(self) -> None:
        """
        Cleanup logic executed before window closure.
        """
        self.stop_webcam_thread()
        if hasattr(self, "webcam") and self.webcam is not None:
            self.webcam.close()
        super().on_close()

    def close(self) -> None:
        """
        Stop acquisition, release camera resources, and unregister module.
        """
        self.on_close()
        super().close()


EXPORTED_CLASS = Webcam_win
EXPORTED_NAME = "Webcam"
