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
from .seecam import Seecam

_ROTATION_MAP = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


class Seecam_win(WindowBase):
    """
    Window module for controlling e-con Systems See3CAM cameras via DirectShow.

    Features:
        - Dynamic format and resolution selection (Y8, Y12, Y16).
        - Adaptive depth conversion from raw 10-bit / 12-bit / 16-bit to 8-bit.
        - Dual-output streaming: 8-bit BGR/RGB and high-depth raw grayscale (FRAME16).
        - Hardware UVC controls: Gain, Exposure, and auto-exposure loop with live peak intensity feedback.
        - Continuous acquisition and software-triggered capture modes.
        - Frame orientation / rotation via global configuration (`Cameras.Seecam.rotation`).
        - Remote automation control via dictionary commands (`IOTypes.CMD_DICT`).
    """

    label: str = "Seecam"
    win_width: int = display_scaling.scale(340)
    win_height: int = display_scaling.scale(490)
    description: str = (
        "Control e-con Systems See3CAM cameras. Supports Y8/Y12/Y16 formats, dual 8-bit/16-bit output, and UVC controls."
    )

    accepted_input_types: list[str] = [IOTypes.CMD_DICT]
    outputs: dict[str, str] = {
        "8bit RGB": IOTypes.FRAME,
        "16bit gray": IOTypes.FRAME16,
    }
    persistent_fields: list[str] = ["label", "cam_name", "acquisition_mode", "depth_mode"]

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the See3CAM Camera window module.

        Args:
            **kwargs: Additional keyword arguments forwarded to WindowBase.
        """
        super().__init__(**kwargs)

        uid = self.UUID
        self.refresh_tag: str = f"seecam_refresh_{uid}"
        self.camlist_tag: str = f"seecam_camlist_{uid}"
        self.camstatus_tag: str = f"seecam_status_{uid}"
        self.cammodel_tag: str = f"seecam_model_{uid}"
        self.caminfo_tag: str = f"seecam_info_{uid}"
        self.start_btn_tag: str = f"seecam_start_{uid}"
        self.stop_btn_tag: str = f"seecam_stop_{uid}"
        self.mode_radio_tag: str = f"seecam_mode_radio_{uid}"
        self.trigger_btn_tag: str = f"seecam_trigger_btn_{uid}"
        self.webcam_gain_tag: str = f"seecam_gain_{uid}"
        self.webcam_exposure_tag: str = f"seecam_exposure_{uid}"
        self.frame_avg_tag: str = f"seecam_frame_avg_{uid}"
        self.auto_exposure_tag: str = f"seecam_auto_exposure_{uid}"
        self.res_list_tag: str = f"seecam_res_list_{uid}"

        self.camlist: list[str] = []
        self.cam_running: bool = False
        self.cam_thread: Optional[threading.Thread] = None

        self.cam_name: str = str(getattr(self, "cam_name", None) or "")
        self.acquisition_mode: str = str(getattr(self, "acquisition_mode", None) or "continuous")
        self.trigger_requested: bool = False
        self.depth_mode: str = str(getattr(self, "depth_mode", None) or "Auto")
        self.auto_exposure_enabled: bool = False

        angle = config.get("Cameras", {}).get("Seecam", {}).get("rotation", 0)
        self._rotation_code: Optional[int] = _ROTATION_MAP.get(angle)

        self._last_frame_lock: threading.Lock = threading.Lock()
        self._last_raw_frame: Optional[np.ndarray] = None
        self._last_8bit_frame: Optional[np.ndarray] = None
        self._last_bgr_frame: Optional[np.ndarray] = None

        self.seecam: Seecam = Seecam()

        self._build_ui()
        self.update_permission()

    def _build_ui(self) -> None:
        """
        Construct DearPyGui widgets for device selection, stream controls, and hardware sliders.
        """
        scale = display_scaling.scale
        with self.window():
            # Device selection
            with dpg.group(horizontal=True):
                dpg.add_button(label="Refresh", tag=self.refresh_tag, callback=self.refresh_cam)
                dpg.add_combo(items=self.camlist, tag=self.camlist_tag, width=-1, callback=self.select_cam)

            dpg.add_text("Status: Not initialized", tag=self.camstatus_tag)
            dpg.add_text("Model: NA", tag=self.cammodel_tag, color=(160, 210, 255))
            dpg.add_text("Info: NA", tag=self.caminfo_tag)

            dpg.add_separator()

            # Acquisition Section: Start / Stop thread and continuous/trigger mode
            dpg.add_text("Acquisition", color=(180, 180, 100))
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Start",
                    tag=self.start_btn_tag,
                    callback=self.start_cam_thread,
                    width=scale(65),
                    enabled=True,
                )
                dpg.add_button(
                    label="Stop",
                    tag=self.stop_btn_tag,
                    callback=self.stop_cam_thread,
                    width=scale(65),
                    enabled=False,
                )
                dpg.add_text("Mode:")
                dpg.add_combo(
                    items=["Continuous", "Trigger"],
                    default_value=self.acquisition_mode.capitalize(),
                    tag=self.mode_radio_tag,
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

            # Camera Hardware Controls (Gain, Exposure, Auto-exp)
            dpg.add_text("Hardware Controls", color=(150, 200, 150))
            with dpg.group(horizontal=True):
                dpg.add_text("Gain: ")
                dpg.add_slider_int(
                    default_value=0,
                    min_value=0,
                    max_value=100,
                    tag=self.webcam_gain_tag,
                    callback=self.set_gain,
                    width=-1,
                )

            with dpg.group(horizontal=True):
                dpg.add_text("Exp:  ")
                dpg.add_slider_int(
                    default_value=0,
                    min_value=-13,
                    max_value=0,
                    tag=self.webcam_exposure_tag,
                    callback=self.set_exposure,
                    width=-1,
                )

            with dpg.group(horizontal=True):
                dpg.add_text("Auto exp:")
                dpg.add_checkbox(
                    default_value=False,
                    tag=self.auto_exposure_tag,
                    callback=lambda s, v: setattr(self, "auto_exposure_enabled", bool(v)),
                )
                dpg.add_spacer(width=scale(15))
                dpg.add_text("Frame max: 0", tag=self.frame_avg_tag)

            dpg.add_separator()

            # Direct Resolution & Capabilities selection
            dpg.add_text("Resolution & Capabilities", color=(150, 150, 200))
            dpg.add_combo(
                items=[],
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

    def refresh_cam(
        self,
        sender: Any = None,
        app_data: Any = None,
        user_data: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Query connected DirectShow video capture devices and update the camera dropdown list.
        """
        self.camlist = self.seecam.list_devices()
        sorted_camlist = sorted(self.camlist, key=lambda c: (0 if "see3" in c.lower() else 1, c))
        if dpg.does_item_exist(self.camlist_tag):
            dpg.configure_item(self.camlist_tag, items=sorted_camlist)
            current_val = dpg.get_value(self.camlist_tag)

            if self.cam_name and self.cam_name in sorted_camlist:
                target = self.cam_name
            elif current_val and current_val in sorted_camlist:
                target = current_val
            elif sorted_camlist:
                target = next((c for c in sorted_camlist if "see3" in c.lower()), sorted_camlist[0])
            else:
                target = ""

            if target:
                dpg.set_value(self.camlist_tag, target)
                self.cam_name = target

    def auto_select_cam(self) -> None:
        """
        Automatically select and initialize the first available See3CAM device found.
        """
        self.refresh_cam()
        if not self.camlist:
            return
        target = next((c for c in self.camlist if "see3" in c.lower()), self.camlist[0])
        if dpg.does_item_exist(self.camlist_tag):
            dpg.set_value(self.camlist_tag, target)
        self.select_cam(cam_name=target)

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
        Initialize the selected camera, update slider limits, and start frame acquisition.

        Args:
            sender: DPG sender tag (if called as callback).
            app_data: DPG app_data value.
            user_data: DPG user data (if applicable).
            cam_name: Name of the DirectShow capture device.
        """
        target = cam_name if cam_name else (app_data if isinstance(app_data, str) else "")
        if not target:
            return

        self.cam_name = target
        self.stop_cam_thread()
        ret, info = self.seecam.init(target)
        self.cam_running = ret

        if ret and isinstance(info, tuple):
            fmt, w, h, th_fps, model_desc = info

            if dpg.does_item_exist(self.camstatus_tag):
                dpg.set_value(self.camstatus_tag, "Status: Running")
            if dpg.does_item_exist(self.cammodel_tag):
                dpg.set_value(self.cammodel_tag, f"Model: {model_desc}")
            if dpg.does_item_exist(self.caminfo_tag):
                dpg.set_value(self.caminfo_tag, f"Format: {fmt} {w}x{h} | Th. {th_fps:.0f} FPS")

            self._update_resolution_dropdown()

            gain_limits = self.seecam.get_gain_limits()
            if gain_limits:
                min_g, max_g = gain_limits[0], gain_limits[1]
                if dpg.does_item_exist(self.webcam_gain_tag):
                    dpg.configure_item(self.webcam_gain_tag, min_value=min_g, max_value=max_g)
                cur_g = self.seecam.get_gain()
                if cur_g is not None and dpg.does_item_exist(self.webcam_gain_tag):
                    dpg.set_value(self.webcam_gain_tag, cur_g)

            exp_limits = self.seecam.get_exposure_limits()
            if exp_limits:
                min_e, max_e = exp_limits[0], exp_limits[1]
                if dpg.does_item_exist(self.webcam_exposure_tag):
                    dpg.configure_item(self.webcam_exposure_tag, min_value=min_e, max_value=max_e)
                cur_e = self.seecam.get_exposure()
                if cur_e is not None and dpg.does_item_exist(self.webcam_exposure_tag):
                    dpg.set_value(self.webcam_exposure_tag, cur_e)

            self.start_cam_thread()
        else:
            if dpg.does_item_exist(self.camstatus_tag):
                dpg.set_value(self.camstatus_tag, "Status: Initialization Error")

    def _update_resolution_dropdown(self) -> None:
        """
        Update the resolutions combo directly with all available formats discovered from DirectShow.
        """
        formats = self.seecam.available_formats
        labels = [f["label"] for f in formats]
        if dpg.does_item_exist(self.res_list_tag):
            dpg.configure_item(self.res_list_tag, items=labels)

            cur_idx = self.seecam.active_format_index
            active_label = next((f["label"] for f in formats if f.get("index") == cur_idx), "")
            if not active_label and labels:
                active_label = labels[0]

            if active_label:
                dpg.set_value(self.res_list_tag, active_label)

    def change_resolution(
        self,
        sender: Any = None,
        app_data: Any = None,
        user_data: Any = None,
        *args: Any,
        res_label: str = "",
        **kwargs: Any,
    ) -> None:
        """
        Change the camera format and resolution dynamically.

        Args:
            sender: DPG sender tag (if called as callback).
            app_data: DPG app_data value.
            user_data: DPG user data (if applicable).
            res_label: Format and resolution string matching an available format item.
        """
        target_label = res_label if res_label else (app_data if isinstance(app_data, str) else "")
        if not target_label:
            return

        target_fmt = next((f for f in self.seecam.available_formats if f.get("label") == target_label), None)
        if not target_fmt:
            return

        format_index = target_fmt["index"]
        if format_index == self.seecam.active_format_index and self.cam_running:
            return

        logger.info(f"[{self.label}] Switching Seecam format to index {format_index}: {target_label}")

        self.stop_cam_thread()

        ret, info = self.seecam.init(self.cam_name, format_index=format_index)
        self.cam_running = ret

        if ret and isinstance(info, tuple):
            fmt, w, h, th_fps, model_desc = info
            if dpg.does_item_exist(self.caminfo_tag):
                dpg.set_value(self.caminfo_tag, f"Format: {fmt} {w}x{h} | Th. {th_fps:.0f} FPS")

            gain_limits = self.seecam.get_gain_limits()
            if gain_limits and dpg.does_item_exist(self.webcam_gain_tag):
                dpg.configure_item(self.webcam_gain_tag, min_value=gain_limits[0], max_value=gain_limits[1])
            exp_limits = self.seecam.get_exposure_limits()
            if exp_limits and dpg.does_item_exist(self.webcam_exposure_tag):
                dpg.configure_item(self.webcam_exposure_tag, min_value=exp_limits[0], max_value=exp_limits[1])

            self.start_cam_thread()
        else:
            logger.error(f"[{self.label}] Failed to switch to format {target_label}")

    def set_gain(
        self,
        sender: Any = None,
        app_data: Any = None,
        user_data: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Set camera hardware gain value."""
        if app_data is not None:
            self.seecam.set_gain(int(app_data))

    def set_exposure(
        self,
        sender: Any = None,
        app_data: Any = None,
        user_data: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Set camera hardware exposure value."""
        if app_data is not None:
            self.seecam.set_exposure(int(app_data))

    def _on_mode_changed(
        self,
        sender: Any = None,
        app_data: Any = "Continuous",
        user_data: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Switch between Continuous streaming and software Trigger mode."""
        mode_val = str(app_data).strip().capitalize()
        if mode_val == "Continuous":
            self.acquisition_mode = "continuous"
            if dpg.does_item_exist(self.trigger_btn_tag):
                dpg.configure_item(self.trigger_btn_tag, enabled=False)
        elif mode_val == "Trigger":
            self.acquisition_mode = "trigger"
            if dpg.does_item_exist(self.trigger_btn_tag):
                dpg.configure_item(self.trigger_btn_tag, enabled=True)

    def _on_trigger(
        self,
        sender: Any = None,
        app_data: Any = None,
        user_data: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Request grab and dispatch of the next frame in Trigger mode."""
        self.trigger_requested = True

    def stop_cam_thread(
        self,
        sender: Any = None,
        app_data: Any = None,
        user_data: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Stop the frame acquisition worker thread."""
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

    def start_cam_thread(
        self,
        sender: Any = None,
        app_data: Any = None,
        user_data: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Start the background frame grabber and FPS tracking worker thread."""
        if not self.seecam.defined:
            target = dpg.get_value(self.camlist_tag) if dpg.does_item_exist(self.camlist_tag) else self.cam_name
            if not target and self.camlist:
                target = self.camlist[0]
            if target:
                self.select_cam(cam_name=target)
                return
            logger.warning(f"[{self.label}] No camera selected to start")
            return

        if self.cam_thread and self.cam_thread.is_alive():
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
                should_grab = False
                if self.acquisition_mode == "continuous":
                    should_grab = True
                elif self.acquisition_mode == "trigger":
                    if self.trigger_requested:
                        self.trigger_requested = False
                        should_grab = True
                    else:
                        time.sleep(0.01)
                        continue

                if not should_grab:
                    time.sleep(0.01)
                    continue

                ret, raw_frame = self.seecam.read()
                if ret and raw_frame is not None:
                    if self.acquisition_mode == "continuous":
                        fps_count += 1
                        now = time.perf_counter()
                        if now - fps_start >= 1.0:
                            live_fps = fps_count / (now - fps_start)
                            fps_count = 0
                            fps_start = now

                            if dpg.does_item_exist(self.caminfo_tag):
                                fmt = self.seecam.format
                                w, h = self.seecam.width, self.seecam.height
                                th = self.seecam.fps
                                dpg.set_value(
                                    self.caminfo_tag,
                                    f"Format: {fmt} {w}x{h} | Th. {th:.0f} FPS | Live: {live_fps:.1f} FPS",
                                )

                    raw_frame = self._apply_rotation(raw_frame)
                    frame8_bit = self.seecam.convert_to_8bit(raw_frame, depth_mode=self.depth_mode)
                    has_bgr_conn = bool(self.connections.get("8bit RGB"))
                    frame_bgr = self.seecam.convert_to_BGR(frame8_bit) if has_bgr_conn else None

                    with self._last_frame_lock:
                        self._last_raw_frame = raw_frame
                        self._last_8bit_frame = frame8_bit
                        self._last_bgr_frame = frame_bgr

                    self.trigger_cb(raw_frame, frame_bgr if frame_bgr is not None else frame8_bit)

                    peak_val = self.seecam.calc_exposure(frame8_bit)
                    if self.auto_exposure_enabled:
                        new_exp = self.seecam.auto_expos(frame8_bit, frame_exposure=peak_val)
                        if new_exp is not None and dpg.does_item_exist(self.webcam_exposure_tag):
                            dpg.set_value(self.webcam_exposure_tag, new_exp)

                    if dpg.does_item_exist(self.frame_avg_tag):
                        dpg.set_value(
                            self.frame_avg_tag,
                            f"Frame max: {peak_val:.0f}",
                        )
                else:
                    time.sleep(0.005)

        self.cam_thread = threading.Thread(
            target=task,
            daemon=True,
            name=f"SeecamWorker_{self.UUID}",
        )
        self.cam_thread.start()

    def _apply_rotation(self, frame: np.ndarray) -> np.ndarray:
        """
        Rotate frame based on camera configuration (Cameras.Seecam.rotation).

        Args:
            frame: Input image array.

        Returns:
            np.ndarray: Rotated image array.
        """
        if self._rotation_code is not None:
            return cv2.rotate(frame, self._rotation_code)
        return frame

    def trigger_cb(self, raw_frame: np.ndarray, bgr_frame: np.ndarray) -> None:
        """
        Emit frames to downstream connected modules (both 8-bit color and 16-bit raw gray).

        Args:
            raw_frame: High bit-depth or raw image array.
            bgr_frame: Standard 8-bit BGR image array.
        """
        rgb_modules = self.connections.get("8bit RGB")
        if rgb_modules:
            if bgr_frame.ndim == 2:
                bgr_frame = self.seecam.convert_to_BGR(bgr_frame)
            source_bgr = (bgr_frame, f"Seecam_{self.cam_name}")
            for module in list(rgb_modules):
                try:
                    module.input_cb(data=source_bgr, data_type=IOTypes.FRAME)
                except Exception as e:
                    logger.error(
                        f"[{self.label}] Error sending frame to {getattr(module, 'label', 'module')} (8bit RGB): {e}"
                    )

        gray_modules = self.connections.get("16bit gray")
        if gray_modules:
            source_raw = (raw_frame, f"Seecam_raw_{self.cam_name}")
            dtype = IOTypes.FRAME16 if raw_frame.dtype == np.uint16 else IOTypes.FRAME
            for module in list(gray_modules):
                try:
                    module.input_cb(data=source_raw, data_type=dtype)
                except Exception as e:
                    logger.error(
                        f"[{self.label}] Error sending frame to {getattr(module, 'label', 'module')} (16bit gray): {e}"
                    )

    def input_cb(self, data: Any = None, *args: Any, **kwargs: Any) -> None:
        """
        Handle incoming command dictionary from connected modules or automation scripts.

        Supported command keys:
            - trigger (bool): Trigger a single frame capture.
            - trigger_cam (bool): Trigger a single frame capture.
            - start (bool): Start continuous acquisition.
            - stop (bool): Stop continuous acquisition.
            - mode (str): Set acquisition mode ("Continuous" or "Trigger").
            - gain (int): Set hardware gain value.
            - exposure (int): Set hardware exposure value.
            - auto_exposure (bool): Enable or disable auto-exposure.
            - depth_mode (str): Set depth conversion mode ("Auto", "Normalize", "8-bit", etc.).
            - resolution (str): Switch format/resolution to a matching label.
            - camera / select_cam (str): Switch active camera device.
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
            self.start_cam_thread()

        if cmd.get("stop"):
            self.stop_cam_thread()

        if "mode" in cmd:
            mode_str = str(cmd["mode"]).strip().capitalize()
            if mode_str in ("Continuous", "Trigger"):
                self._on_mode_changed(app_data=mode_str)
                if dpg.does_item_exist(self.mode_radio_tag):
                    dpg.set_value(self.mode_radio_tag, mode_str)

        if "gain" in cmd:
            try:
                val_gain = int(cmd["gain"])
                self.set_gain(app_data=val_gain)
                if dpg.does_item_exist(self.webcam_gain_tag):
                    dpg.set_value(self.webcam_gain_tag, val_gain)
            except (ValueError, TypeError) as e:
                logger.warning(f"[{self.label}] Invalid gain value '{cmd['gain']}': {e}")

        if "exposure" in cmd:
            try:
                val_exp = int(cmd["exposure"])
                self.set_exposure(app_data=val_exp)
                if dpg.does_item_exist(self.webcam_exposure_tag):
                    dpg.set_value(self.webcam_exposure_tag, val_exp)
            except (ValueError, TypeError) as e:
                logger.warning(f"[{self.label}] Invalid exposure value '{cmd['exposure']}': {e}")

        if "auto_exposure" in cmd:
            val_auto = bool(cmd["auto_exposure"])
            self.auto_exposure_enabled = val_auto
            if dpg.does_item_exist(self.auto_exposure_tag):
                dpg.set_value(self.auto_exposure_tag, val_auto)

        if "depth_mode" in cmd:
            self.depth_mode = str(cmd["depth_mode"])

        if "resolution" in cmd:
            self.change_resolution(res_label=str(cmd["resolution"]))

        if "camera" in cmd or "select_cam" in cmd:
            target_cam = str(cmd.get("camera") or cmd.get("select_cam"))
            self.select_cam(cam_name=target_cam)

        if cmd.get("trigger_cam") or cmd.get("trigger"):
            self._on_trigger()

    def get_last_frames(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Thread-safe access to the most recently captured raw and BGR frames.

        Returns:
            Tuple containing (raw_frame, bgr_frame).
        """
        with self._last_frame_lock:
            raw_copy = self._last_raw_frame.copy() if self._last_raw_frame is not None else None
            if self._last_bgr_frame is not None:
                bgr_copy = self._last_bgr_frame.copy()
            elif self._last_8bit_frame is not None:
                bgr_copy = self.seecam.convert_to_BGR(self._last_8bit_frame)
            else:
                bgr_copy = None
            return raw_copy, bgr_copy

    def is_ready(self) -> bool:
        """
        Check if the See3CAM camera module is ready to accept commands or frames.

        Returns:
            bool: Always True.
        """
        return True

    def on_close(self) -> None:
        """
        Cleanup logic executed before window closure.
        """
        self.stop_cam_thread()
        if hasattr(self, "seecam") and self.seecam is not None:
            self.seecam.close()
        super().on_close()

    def close(self) -> None:
        """
        Stop acquisition, release camera resources, and unregister module.
        """
        self.on_close()
        super().close()


EXPORTED_CLASS = Seecam_win
EXPORTED_NAME = "Seecam"
