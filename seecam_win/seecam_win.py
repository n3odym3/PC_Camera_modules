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
from .seecam import Seecam


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

    description: str = "Control e-con Systems See3CAM cameras. Supports Y8/Y12/Y16 formats, dual 8-bit/16-bit output, and UVC controls."

    refresh_tag: str
    camlist_tag: str
    camstatus_tag: str
    cammodel_tag: str
    caminfo_tag: str
    start_btn_tag: str
    stop_btn_tag: str
    mode_radio_tag: str
    trigger_btn_tag: str
    webcam_gain_tag: str
    webcam_exposure_tag: str
    frame_avg_tag: str
    auto_exposure_tag: str
    res_list_tag: str

    camlist: List[str]
    cam_running: bool
    cam_thread: Optional[threading.Thread]

    cam_name: str
    acquisition_mode: str
    trigger_requested: bool
    depth_mode: str

    _last_frame_lock: threading.Lock
    _last_raw_frame: Optional[np.ndarray]
    _last_bgr_frame: Optional[np.ndarray]

    seecam: Seecam

    def __init__(
        self,
        label: str = "Seecam",
        win_width: int = display_scaling.scale(340),
        win_height: int = display_scaling.scale(490),
        pos: Tuple[int, int] = (10, 10),
        uuid: Optional[str] = None,
        outputs: Optional[Dict[str, Any]] = None,
        visible: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the See3CAM Camera window module.

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

        self.refresh_tag = f"seecam_refresh_{self.UUID}"
        self.camlist_tag = f"seecam_camlist_{self.UUID}"
        self.camstatus_tag = f"seecam_status_{self.UUID}"
        self.cammodel_tag = f"seecam_model_{self.UUID}"
        self.caminfo_tag = f"seecam_info_{self.UUID}"
        self.start_btn_tag = f"seecam_start_{self.UUID}"
        self.stop_btn_tag = f"seecam_stop_{self.UUID}"
        self.mode_radio_tag = f"seecam_mode_radio_{self.UUID}"
        self.trigger_btn_tag = f"seecam_trigger_btn_{self.UUID}"
        self.webcam_gain_tag = f"seecam_gain_{self.UUID}"
        self.webcam_exposure_tag = f"seecam_exposure_{self.UUID}"
        self.frame_avg_tag = f"seecam_frame_avg_{self.UUID}"
        self.auto_exposure_tag = f"seecam_auto_exposure_{self.UUID}"
        self.res_list_tag = f"seecam_res_list_{self.UUID}"

        self.camlist = []
        self.cam_running = False
        self.cam_thread = None

        self.cam_name = ""
        self.acquisition_mode = "continuous"
        self.trigger_requested = False
        self.depth_mode = "Auto"

        self._last_frame_lock = threading.Lock()
        self._last_raw_frame = None
        self._last_bgr_frame = None

        self.seecam = Seecam()

        self._persistent_fields = ["label"]
        self.accepted_input_types = [IOTypes.CMD_DICT]
        self.outputs = {
            "8bit RGB": IOTypes.FRAME,
            "16bit gray": IOTypes.FRAME16,
        }
        self.connections = {k: [] for k in self.outputs}

        self._build_ui()
        self.update_permission()

    def _build_ui(self) -> None:
        """
        Construct DearPyGui widgets for device selection, stream controls, and hardware sliders.
        """
        with dpg.window(
            label=self.label,
            width=self.win_width,
            height=self.win_height,
            pos=self.pos,
            tag=self.winID,
            show=self.visible,
        ):
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
                    width=display_scaling.scale(65),
                    enabled=True,
                )
                dpg.add_button(
                    label="Stop",
                    tag=self.stop_btn_tag,
                    callback=self.stop_cam_thread,
                    width=display_scaling.scale(65),
                    enabled=False,
                )
                dpg.add_text("Mode:")
                dpg.add_combo(
                    items=["Continuous", "Trigger"],
                    default_value="Continuous",
                    tag=self.mode_radio_tag,
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
                dpg.add_checkbox(default_value=False, tag=self.auto_exposure_tag)
                dpg.add_spacer(width=display_scaling.scale(15))
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

    def refresh_cam(self, *args: Any, **kwargs: Any) -> None:
        """
        Query connected DirectShow video capture devices and update the camera dropdown list.
        """
        self.camlist = self.seecam.list_devices()
        sorted_camlist = sorted(self.camlist, key=lambda c: (0 if "see3" in c.lower() else 1, c))
        dpg.configure_item(self.camlist_tag, items=sorted_camlist)
        if sorted_camlist and dpg.does_item_exist(self.camlist_tag):
            current_val = dpg.get_value(self.camlist_tag)
            if not current_val:
                target = None
                for c in sorted_camlist:
                    if "see3" in c.lower():
                        target = c
                        break
                if not target:
                    target = sorted_camlist[0]
                dpg.set_value(self.camlist_tag, target)
                self.cam_name = target

    def auto_select_cam(self) -> None:
        """
        Automatically select and initialize the first available See3CAM device found.
        """
        self.refresh_cam()
        target = None
        for camname in self.camlist:
            if "see3" in camname.lower():
                target = camname
                break
        if not target and self.camlist:
            target = self.camlist[0]

        if target:
            dpg.set_value(self.camlist_tag, target)
            self.select_cam(None, target)

    def select_cam(self, sender: Any, cam_name: str, *args: Any, **kwargs: Any) -> None:
        """
        Initialize the selected camera, update slider limits, and start frame acquisition.

        Args:
            cam_name: Name of the DirectShow capture device.
        """
        if not cam_name:
            return

        self.cam_name = cam_name
        self.stop_cam_thread()
        ret, info = self.seecam.init(cam_name)
        self.cam_running = ret

        if ret and isinstance(info, tuple):
            fmt, w, h, th_fps, model_desc = info

            if dpg.does_item_exist(self.camstatus_tag):
                dpg.set_value(self.camstatus_tag, "Status: Running")
            if dpg.does_item_exist(self.cammodel_tag):
                dpg.set_value(self.cammodel_tag, f"Model: {model_desc}")
            if dpg.does_item_exist(self.caminfo_tag):
                dpg.set_value(self.caminfo_tag, f"Format: {fmt} {w}x{h} | Th. {th_fps:.0f} FPS")

            # Update stream capabilities dropdown directly
            self._update_resolution_dropdown()

            # Configure Gain Slider limits
            gain_limits = self.seecam.get_gain_limits()
            if gain_limits:
                min_g, max_g = gain_limits[0], gain_limits[1]
                dpg.configure_item(self.webcam_gain_tag, min_value=min_g, max_value=max_g)
                cur_g = self.seecam.get_gain()
                if cur_g is not None:
                    dpg.set_value(self.webcam_gain_tag, cur_g)

            # Configure Exposure Slider limits
            exp_limits = self.seecam.get_exposure_limits()
            if exp_limits:
                min_e, max_e = exp_limits[0], exp_limits[1]
                dpg.configure_item(self.webcam_exposure_tag, min_value=min_e, max_value=max_e)
                cur_e = self.seecam.get_exposure()
                if cur_e is not None:
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
        dpg.configure_item(self.res_list_tag, items=labels)

        active_label = ""
        cur_idx = self.seecam.active_format_index
        for f in formats:
            if f["index"] == cur_idx:
                active_label = f["label"]
                break
        if not active_label and labels:
            active_label = labels[0]

        if active_label:
            dpg.set_value(self.res_list_tag, active_label)

    def change_resolution(self, sender: Any, res_label: str, *args: Any, **kwargs: Any) -> None:
        """
        Change the camera format and resolution dynamically.

        Args:
            res_label: Format and resolution string matching an available format item.
        """
        if not res_label:
            return

        target_fmt = None
        for f in self.seecam.available_formats:
            if f["label"] == res_label:
                target_fmt = f
                break

        if not target_fmt:
            return

        format_index = target_fmt["index"]
        if format_index == self.seecam.active_format_index and self.cam_running:
            return

        logger.info(f"Switching Seecam format to index {format_index}: {res_label}")

        # Stop running acquisition thread during graph rebuild
        self.stop_cam_thread()

        ret, info = self.seecam.init(self.cam_name, format_index=format_index)
        self.cam_running = ret

        if ret and isinstance(info, tuple):
            fmt, w, h, th_fps, model_desc = info
            if dpg.does_item_exist(self.caminfo_tag):
                dpg.set_value(self.caminfo_tag, f"Format: {fmt} {w}x{h} | Th. {th_fps:.0f} FPS")

            gain_limits = self.seecam.get_gain_limits()
            if gain_limits:
                dpg.configure_item(self.webcam_gain_tag, min_value=gain_limits[0], max_value=gain_limits[1])
            exp_limits = self.seecam.get_exposure_limits()
            if exp_limits:
                dpg.configure_item(self.webcam_exposure_tag, min_value=exp_limits[0], max_value=exp_limits[1])

            self.start_cam_thread()
        else:
            logger.error(f"Failed to switch to format {res_label}")

    def set_gain(self, sender: Any, app_data: int, user_data: Any = None, *args: Any) -> None:
        """
        Set camera hardware gain value.
        """
        self.seecam.set_gain(app_data)

    def set_exposure(self, sender: Any, app_data: int, user_data: Any = None, *args: Any) -> None:
        """
        Set camera hardware exposure value.
        """
        self.seecam.set_exposure(app_data)

    def _on_mode_changed(self, sender: Any, app_data: str, user_data: Any = None, *args: Any) -> None:
        """
        Switch between Continuous streaming and software Trigger mode.
        """
        if app_data == "Continuous":
            self.acquisition_mode = "continuous"
            dpg.configure_item(self.trigger_btn_tag, enabled=False)
        elif app_data == "Trigger":
            self.acquisition_mode = "trigger"
            dpg.configure_item(self.trigger_btn_tag, enabled=True)

    def _on_trigger(self, *args: Any, **kwargs: Any) -> None:
        """
        Request grab and dispatch of the next frame in Trigger mode.
        """
        self.trigger_requested = True

    def stop_cam_thread(self, *args: Any, **kwargs: Any) -> None:
        """
        Stop the frame acquisition worker thread.
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

    def start_cam_thread(self, *args: Any, **kwargs: Any) -> None:
        """
        Start the background frame grabber and FPS tracking worker thread.
        """
        if not self.seecam.defined:
            target = dpg.get_value(self.camlist_tag) if dpg.does_item_exist(self.camlist_tag) else self.cam_name
            if not target and self.camlist:
                target = self.camlist[0]
            if target:
                self.select_cam(None, target)
                return
            logger.warning("No camera selected to start")
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
                if self.acquisition_mode == "continuous":
                    ret, raw_frame = self.seecam.read()
                    if ret and raw_frame is not None:
                        fps_count += 1
                        now = time.perf_counter()
                        if now - fps_start >= 1.0:
                            live_fps = fps_count / (now - fps_start)
                            fps_count = 0
                            fps_start = now

                            if dpg.does_item_exist(self.caminfo_tag):
                                fmt, w, h, th = self.seecam.format, self.seecam.width, self.seecam.height, self.seecam.fps
                                dpg.set_value(self.caminfo_tag, f"Format: {fmt} {w}x{h} | Th. {th:.0f} FPS | Live: {live_fps:.1f} FPS")

                        raw_frame = self._apply_rotation(raw_frame)
                        frame8_bit = self.seecam.convert_to_8bit(raw_frame, depth_mode=self.depth_mode)
                        frame_bgr = self.seecam.convert_to_BGR(frame8_bit)

                        with self._last_frame_lock:
                            self._last_raw_frame = raw_frame
                            self._last_bgr_frame = frame_bgr

                        self.trigger_cb(raw_frame, frame_bgr)

                        if dpg.get_value(self.auto_exposure_tag):
                            new_exp = self.seecam.auto_expos(frame8_bit)
                            dpg.set_value(self.webcam_exposure_tag, new_exp)

                        if dpg.does_item_exist(self.frame_avg_tag):
                            dpg.set_value(self.frame_avg_tag, f"Frame max: {self.seecam.calc_exposure(frame8_bit):.0f}")
                    else:
                        time.sleep(0.005)

                elif self.acquisition_mode == "trigger":
                    if self.trigger_requested:
                        self.trigger_requested = False
                        ret, raw_frame = self.seecam.read()
                        if ret and raw_frame is not None:
                            raw_frame = self._apply_rotation(raw_frame)
                            frame8_bit = self.seecam.convert_to_8bit(raw_frame, depth_mode=self.depth_mode)
                            frame_bgr = self.seecam.convert_to_BGR(frame8_bit)

                            with self._last_frame_lock:
                                self._last_raw_frame = raw_frame
                                self._last_bgr_frame = frame_bgr

                            self.trigger_cb(raw_frame, frame_bgr)

                            if dpg.get_value(self.auto_exposure_tag):
                                new_exp = self.seecam.auto_expos(frame8_bit)
                                self.seecam.set_exposure(new_exp)
                                dpg.set_value(self.webcam_exposure_tag, new_exp)

                            if dpg.does_item_exist(self.frame_avg_tag):
                                dpg.set_value(self.frame_avg_tag, f"Frame max: {self.seecam.calc_exposure(frame8_bit):.0f}")
                    else:
                        time.sleep(0.01)

        self.cam_thread = threading.Thread(target=task, daemon=True)
        self.cam_thread.start()

    def _apply_rotation(self, frame: np.ndarray) -> np.ndarray:
        """
        Rotate frame based on camera configuration (Cameras.Seecam.rotation).

        Args:
            frame: Input image array.

        Returns:
            np.ndarray: Rotated image array.
        """
        angle = config.get("Cameras", {}).get("Seecam", {}).get("rotation", 0)
        if angle == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def trigger_cb(self, raw_frame: np.ndarray, bgr_frame: np.ndarray) -> None:
        """
        Emit frames to downstream connected modules (both 8-bit color and 16-bit raw gray).

        Args:
            raw_frame: High bit-depth or raw image array.
            bgr_frame: Standard 8-bit BGR image array.
        """
        source_bgr = (bgr_frame, f"Seecam_{self.cam_name}")
        source_raw = (raw_frame, f"Seecam_raw_{self.cam_name}")

        for idx, output_key in enumerate(self.outputs):
            connected_modules = list(self.connections.get(output_key, []))
            for module in connected_modules:
                try:
                    if idx == 0:  # "8bit RGB"
                        module.input_cb(data=source_bgr, data_type=IOTypes.FRAME)
                    elif idx == 1:  # "16bit gray"
                        dtype = IOTypes.FRAME16 if raw_frame.dtype == np.uint16 else IOTypes.FRAME
                        module.input_cb(data=source_raw, data_type=dtype)
                except Exception as e:
                    logger.error(f"Error sending frame from Seecam ({output_key}): {e}")

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
            - camera (str): Switch active camera device.
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
                self.start_cam_thread()

            if "stop" in cmd and cmd["stop"]:
                self.stop_cam_thread()

            if "mode" in cmd:
                mode_str = str(cmd["mode"]).capitalize()
                if mode_str in ("Continuous", "Trigger"):
                    self._on_mode_changed(None, mode_str)
                    if dpg.does_item_exist(self.mode_radio_tag):
                        dpg.set_value(self.mode_radio_tag, mode_str)

            if "gain" in cmd:
                val_gain = int(cmd["gain"])
                self.set_gain(None, val_gain)
                if dpg.does_item_exist(self.webcam_gain_tag):
                    dpg.set_value(self.webcam_gain_tag, val_gain)

            if "exposure" in cmd:
                val_exp = int(cmd["exposure"])
                self.set_exposure(None, val_exp)
                if dpg.does_item_exist(self.webcam_exposure_tag):
                    dpg.set_value(self.webcam_exposure_tag, val_exp)

            if "auto_exposure" in cmd:
                val_auto = bool(cmd["auto_exposure"])
                if dpg.does_item_exist(self.auto_exposure_tag):
                    dpg.set_value(self.auto_exposure_tag, val_auto)

            if "depth_mode" in cmd:
                self.depth_mode = str(cmd["depth_mode"])

            if "resolution" in cmd:
                self.change_resolution(None, str(cmd["resolution"]))

            if "camera" in cmd or "select_cam" in cmd:
                target_cam = str(cmd.get("camera") or cmd.get("select_cam"))
                self.select_cam(None, target_cam)

            if cmd.get("trigger_cam") or cmd.get("trigger"):
                self._on_trigger()

    def is_ready(self) -> bool:
        """
        Check if the See3CAM camera module is ready to accept commands or frames.

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
        self.stop_cam_thread()
        self.seecam.close()
        super().on_close()

    def close(self) -> None:
        """
        Stop acquisition, release camera resources, and unregister module.
        """
        self.stop_cam_thread()
        self.seecam.close()
        super().close()


EXPORTED_CLASS = Seecam_win
EXPORTED_NAME = "Seecam"
