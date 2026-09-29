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
from core.config_manager import config
from core.app_state import app_state
from .hamamatsu import Hamamatsu


class Hamamatsu_win(WindowBase):
    """
    Window module for controlling Hamamatsu scientific cameras via pyDCAM.

    Features:
        - Continuous live acquisition ("Video"), hardware ("Trigger"), and software ("Trigger Software") modes.
        - 16-bit monochrome frame streaming (output as `IOTypes.FRAME16`).
        - Real-time dark reference accumulation and background subtraction with dynamic offset.
        - Area of Interest (AOI) / subarray configuration with automatic buffer reallocation.
        - Hardware binning (1x1, 2x2, 4x4) and readout speed selection (Fastest, Slowest).
        - Remote automation control via dictionary commands (`IOTypes.CMD_DICT`).
    """

    description: str = "Control Hamamatsu scientific cameras. Supports exposure, trigger modes, software trigger, and 16-bit frame output."

    init_btn_tag: str
    _suppress_emission: bool
    exposure_tag: str
    trigger_tag: str
    start_btn_tag: str
    width_tag: str
    height_tag: str
    binning_tag: str
    readout_speed_tag: str

    cam_running: bool
    cam_thread: Optional[threading.Thread]
    hamamatsu_cam: Hamamatsu

    dark_ref_frame: Optional[np.ndarray]
    collecting_dark_ref: bool
    dark_ref_n_images: int
    dark_ref_accumulator: Optional[np.ndarray]
    dark_ref_counter: int
    correct_dark: bool
    dark_threshold: int

    width: Optional[int]
    height: Optional[int]
    binning: int
    readout_speed: str

    def __init__(
        self,
        label: str = "Hamamatsu Camera",
        win_width: int = 350,
        win_height: int = 300,
        pos: Tuple[int, int] = (10, 10),
        uuid: Optional[str] = None,
        outputs: Optional[Dict[str, Any]] = None,
        visible: bool = True,
        width: Optional[int] = None,
        height: Optional[int] = None,
        binning: Optional[int] = None,
        readout_speed: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the Hamamatsu Camera window module.

        Args:
            label: Window title displayed in the UI.
            win_width: Initial window width in pixels.
            win_height: Initial window height in pixels.
            pos: Initial (x, y) window position on screen.
            uuid: Optional unique module identifier.
            outputs: Optional predefined output connections dictionary.
            visible: Whether the window is visible upon creation.
            width: Optional pre-configured sensor AOI width.
            height: Optional pre-configured sensor AOI height.
            binning: Optional pre-configured hardware binning factor (1, 2, or 4).
            readout_speed: Optional pre-configured readout speed ("Fastest" or "Slowest").
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
        )

        self.init_btn_tag = f"hama_init_btn_{self.UUID}"
        self._suppress_emission = False
        self.exposure_tag = f"hama_exposure_{self.UUID}"
        self.trigger_tag = f"hama_trigger_{self.UUID}"
        self.start_btn_tag = f"hama_start_btn_{self.UUID}"
        self.width_tag = f"hama_width_{self.UUID}"
        self.height_tag = f"hama_height_{self.UUID}"
        self.binning_tag = f"hama_binning_{self.UUID}"
        self.readout_speed_tag = f"hama_readout_speed_{self.UUID}"

        self.cam_running = False
        self.cam_thread = None
        self.hamamatsu_cam = Hamamatsu()

        self.dark_ref_frame = None
        self.collecting_dark_ref = False
        self.dark_ref_n_images = 10
        self.dark_ref_accumulator = None
        self.dark_ref_counter = 0
        self.correct_dark = True
        self.dark_threshold = 10

        self._persistent_fields = ["label", "width", "height", "binning", "readout_speed"]
        self.binning = binning if binning is not None else 1
        self.readout_speed = readout_speed if readout_speed is not None else "Fastest"
        self.width = width
        self.height = height
        self.accepted_input_types = [IOTypes.CMD_DICT]
        self.outputs = {
            "16bit frame": IOTypes.FRAME16,
        }
        self.connections = {k: [] for k in self.outputs}

        if app_state.get("detector") is None:
            app_state.detector = 50000

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
                    label="Initialize",
                    tag=self.init_btn_tag,
                    callback=self._init_camera,
                )
                dpg.add_button(
                    label="Start",
                    tag=self.start_btn_tag,
                    callback=self._start_capture,
                )
                dpg.add_button(label="Stop", callback=self._stop_capture)

            dpg.add_separator()

            with dpg.group(horizontal=True):
                dpg.add_combo(
                    items=["Video", "Trigger", "Trigger Software"],
                    default_value="Trigger",
                    tag=self.trigger_tag,
                    callback=self._set_trigger,
                    width=150,
                )
                dpg.add_button(
                    label="Trigger",
                    callback=lambda *args: self.hamamatsu_cam.trigger(),
                )

            with dpg.group(horizontal=True):
                dpg.add_drag_int(
                    label="Expos. (µs)",
                    default_value=50000,
                    min_value=1000,
                    max_value=10000000,
                    tag=self.exposure_tag,
                    callback=self._set_exposure,
                    width=100,
                )
                dpg.add_combo(
                    items=["1x1", "2x2", "4x4"],
                    default_value=f"{self.binning}x{self.binning}",
                    tag=self.binning_tag,
                    callback=self._set_binning,
                    width=80,
                )
                dpg.add_combo(
                    items=["Slowest", "Fastest"],
                    default_value=self.readout_speed,
                    tag=self.readout_speed_tag,
                    callback=self._set_readout_speed,
                    width=80,
                )

            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Ref frame", callback=self._start_dark_ref_capture
                )
                dpg.add_button(label="Clear", callback=self._clear_dark_ref)

            with dpg.collapsing_header(label="Advanced"):
                dpg.add_input_int(
                    label="N Ref rep",
                    default_value=self.dark_ref_n_images,
                    width=100,
                    step=-1,
                    callback=lambda s, a, *args: setattr(
                        self, "dark_ref_n_images", a
                    ),
                )

                dpg.add_checkbox(
                    label="Correct dark",
                    default_value=self.correct_dark,
                    callback=lambda s, a, *args: setattr(
                        self, "correct_dark", a
                    ),
                )

                dpg.add_input_int(
                    label="Dark threshold",
                    default_value=self.dark_threshold,
                    width=100,
                    step=-1,
                    callback=lambda s, a, *args: setattr(
                        self, "dark_threshold", a
                    ),
                )

                dpg.add_separator()
                dpg.add_text("AOI / Resolution")
                with dpg.group(horizontal=True):
                    dpg.add_input_int(
                        label="Width",
                        default_value=self.width if self.width is not None else 2048,
                        tag=self.width_tag,
                        width=100,
                        step=0,
                        callback=lambda s, a: setattr(self, "width", a),
                    )
                    dpg.add_input_int(
                        label="Height",
                        default_value=self.height if self.height is not None else 2048,
                        tag=self.height_tag,
                        width=100,
                        step=0,
                        callback=lambda s, a: setattr(self, "height", a),
                    )
                dpg.add_button(
                    label="Update Resolution", callback=self._apply_aoi
                )

        self.autosize_window()
        self.update_permission()

    def update_permission(self) -> None:
        """
        Adjust module permissions and UI visibility based on the application mode.
        """
        is_user = app_state.mode == "user"

        if dpg.does_item_exist(self.winID):
            dpg.configure_item(self.winID, no_close=is_user)

    def _init_camera(
        self,
        sender: Optional[str] = None,
        app_data: Any = None,
        user_data: Any = None,
        *args: Any,
    ) -> None:
        """
        Asynchronously initialize the Hamamatsu camera hardware, configure
        initial exposure, trigger, binning, and readout parameters, and start continuous capture.
        """
        with dpg.theme() as theme_init:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(
                    dpg.mvThemeCol_Button,
                    (255, 165, 0),
                    category=dpg.mvThemeCat_Core,
                )
                dpg.add_theme_color(
                    dpg.mvThemeCol_ButtonHovered,
                    (255, 180, 50),
                    category=dpg.mvThemeCat_Core,
                )
                dpg.add_theme_color(
                    dpg.mvThemeCol_ButtonActive,
                    (255, 140, 0),
                    category=dpg.mvThemeCat_Core,
                )
            dpg.bind_item_theme(self.init_btn_tag, theme_init)

        def init_task() -> None:
            try:
                readout_speed_val = dpg.get_value(self.readout_speed_tag)
                self.hamamatsu_cam.readout_speed = readout_speed_val

                if self.hamamatsu_cam.init():
                    with dpg.theme() as theme_success:
                        with dpg.theme_component(dpg.mvAll):
                            dpg.add_theme_color(
                                dpg.mvThemeCol_Button,
                                (0, 100, 0),
                                category=dpg.mvThemeCat_Core,
                            )
                            dpg.add_theme_color(
                                dpg.mvThemeCol_ButtonHovered,
                                (0, 150, 0),
                                category=dpg.mvThemeCat_Core,
                            )
                            dpg.add_theme_color(
                                dpg.mvThemeCol_ButtonActive,
                                (0, 200, 0),
                                category=dpg.mvThemeCat_Core,
                            )
                    dpg.bind_item_theme(self.init_btn_tag, theme_success)

                    logger.success("Hamamatsu camera initialized")

                    exposure_us = dpg.get_value(self.exposure_tag)
                    self._set_exposure(None, exposure_us)

                    trigger_mode = dpg.get_value(self.trigger_tag)
                    self.hamamatsu_cam.set_trigger(trigger_mode)

                    binning_str = dpg.get_value(self.binning_tag)
                    binning_val = int(binning_str.split("x")[0])
                    res = self.hamamatsu_cam.set_binning(binning_val)
                    if isinstance(res, tuple):
                        self.width, self.height = res
                    else:
                        self.width = int(self.hamamatsu_cam.width)
                        self.height = int(self.hamamatsu_cam.height)

                    if dpg.does_item_exist(self.width_tag):
                        dpg.set_value(self.width_tag, self.width)
                    if dpg.does_item_exist(self.height_tag):
                        dpg.set_value(self.height_tag, self.height)

                    self._start_capture()
                else:
                    with dpg.theme() as theme_fail:
                        with dpg.theme_component(dpg.mvAll):
                            dpg.add_theme_color(
                                dpg.mvThemeCol_Button,
                                (150, 0, 0),
                                category=dpg.mvThemeCat_Core,
                            )
                            dpg.add_theme_color(
                                dpg.mvThemeCol_ButtonHovered,
                                (200, 0, 0),
                                category=dpg.mvThemeCat_Core,
                            )
                            dpg.add_theme_color(
                                dpg.mvThemeCol_ButtonActive,
                                (255, 0, 0),
                                category=dpg.mvThemeCat_Core,
                            )
                    dpg.bind_item_theme(self.init_btn_tag, theme_fail)
                    logger.error("Hamamatsu camera initialization failed")
            except Exception as e:
                with dpg.theme() as theme_error:
                    with dpg.theme_component(dpg.mvAll):
                        dpg.add_theme_color(
                            dpg.mvThemeCol_Button,
                            (150, 0, 0),
                            category=dpg.mvThemeCat_Core,
                        )
                        dpg.add_theme_color(
                            dpg.mvThemeCol_ButtonHovered,
                            (200, 0, 0),
                            category=dpg.mvThemeCat_Core,
                        )
                        dpg.add_theme_color(
                            dpg.mvThemeCol_ButtonActive,
                            (255, 0, 0),
                            category=dpg.mvThemeCat_Core,
                        )
                dpg.bind_item_theme(self.init_btn_tag, theme_error)
                logger.error(f"Hamamatsu init error: {e}")

        threading.Thread(target=init_task, daemon=True).start()

    def _apply_rotation(self, frame: np.ndarray) -> np.ndarray:
        """
        Rotate frame based on camera configuration (Cameras.Hamamatsu.rotation).

        Args:
            frame: Input image array.

        Returns:
            np.ndarray: Rotated image array.
        """
        angle = config.get("Cameras", {}).get("Hamamatsu", {}).get("rotation", 0)
        if angle == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def _start_capture(
        self,
        sender: Optional[str] = None,
        app_data: Any = None,
        user_data: Any = None,
        *args: Any,
    ) -> None:
        """
        Start the background acquisition thread to capture frames from the camera.
        """
        if not self.hamamatsu_cam.defined:
            return

        if self.cam_thread and self.cam_thread.is_alive():
            return

        self.cam_running = True

        with dpg.theme() as theme_active:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(
                    dpg.mvThemeCol_Button,
                    (0, 100, 0),
                    category=dpg.mvThemeCat_Core,
                )
                dpg.add_theme_color(
                    dpg.mvThemeCol_ButtonHovered,
                    (0, 150, 0),
                    category=dpg.mvThemeCat_Core,
                )
                dpg.add_theme_color(
                    dpg.mvThemeCol_ButtonActive,
                    (0, 200, 0),
                    category=dpg.mvThemeCat_Core,
                )
            dpg.bind_item_theme(self.start_btn_tag, theme_active)

        def capture_task() -> None:
            while self.cam_running:
                if not self.hamamatsu_cam.defined or self.hamamatsu_cam.hwait is None:
                    time.sleep(0.05)
                    continue

                has_event = self.hamamatsu_cam.check_event(timeout_ms=100)

                if has_event:
                    frame = self.hamamatsu_cam.get_image()
                    if frame is not None:
                        frame = self._apply_rotation(frame)

                        is_collecting = self.collecting_dark_ref

                        if is_collecting:
                            if self.dark_ref_accumulator is None:
                                self.dark_ref_accumulator = frame.astype(np.float32)
                            else:
                                self.dark_ref_accumulator += frame.astype(np.float32)

                            self.dark_ref_counter += 1

                            if self.dark_ref_counter >= self.dark_ref_n_images:
                                self.dark_ref_frame = self.dark_ref_accumulator / self.dark_ref_counter
                                self.collecting_dark_ref = False
                                logger.success(f"Dark reference captured from {self.dark_ref_counter} frames")

                        ref_frame = self.dark_ref_frame
                        if self.correct_dark and ref_frame is not None and ref_frame.shape == frame.shape:
                            corrected = frame.astype(np.float32) - ref_frame

                            mask = corrected < self.dark_threshold
                            if np.any(mask):
                                dynamic_offset = np.mean(corrected[mask])
                                corrected -= dynamic_offset

                            corrected = np.clip(corrected, 0, 65535)
                            frame = corrected.astype(np.uint16)

                        if not is_collecting and not self._suppress_emission:
                            self._trigger_outputs(frame)

        self.cam_thread = threading.Thread(target=capture_task, daemon=True)
        self.cam_thread.start()

    def _stop_capture(
        self,
        sender: Optional[str] = None,
        app_data: Any = None,
        user_data: Any = None,
        *args: Any,
    ) -> None:
        """
        Stop the background acquisition thread and reset Start button styling.
        """
        if not self.cam_running:
            return
        self.cam_running = False
        if self.cam_thread:
            self.cam_thread.join(timeout=1.0)
        dpg.bind_item_theme(self.start_btn_tag, 0)

    def _trigger_outputs(self, frame: np.ndarray) -> None:
        """
        Dispatch the captured frame tuple (frame, "Hamamatsu") to all connected modules.

        Args:
            frame: 16-bit monochrome frame array (uint16).
        """
        data = (frame, "Hamamatsu")
        for output_key, connected_modules in self.connections.items():
            for module in connected_modules:
                try:
                    module.input_cb(data)
                except Exception as e:
                    logger.error(f"Error sending frame to {module}: {e}")

    def _set_exposure(self, sender: Any, app_data: Any, user_data: Any = None, *args: Any) -> None:
        """
        Set camera exposure time in microseconds and update global detector state.
        """
        if app_data is not None:
            try:
                app_state.detector = int(app_data)
            except (ValueError, TypeError):
                pass
        if self.hamamatsu_cam.defined:
            self.hamamatsu_cam.set_exposure(app_data)

    def _set_trigger(self, sender: Any, app_data: Any, user_data: Any = None, *args: Any) -> None:
        """
        Set acquisition trigger mode ("Video", "Trigger", or "Trigger Software").
        """
        if self.hamamatsu_cam.defined:
            self.hamamatsu_cam.set_trigger(app_data)

    def _set_readout_speed(self, sender: Any, app_data: Any, user_data: Any = None, *args: Any) -> None:
        """
        Set sensor readout speed ("Fastest" or "Slowest").
        """
        self.readout_speed = app_data
        self.dark_ref_frame = None
        if self.hamamatsu_cam.defined:
            was_running = self.cam_running
            if was_running:
                self._stop_capture()
            self.hamamatsu_cam.set_readout_speed(app_data)
            if was_running:
                self._start_capture()

    def _set_binning(self, sender: Any, app_data: Any, user_data: Any = None, *args: Any) -> None:
        """
        Set sensor binning factor (1x1, 2x2, 4x4) and update resolution fields.
        """
        binning_val = int(app_data.split("x")[0])
        self.binning = binning_val
        self.dark_ref_frame = None
        if self.hamamatsu_cam.defined:
            was_running = self.cam_running
            if was_running:
                self._stop_capture()

            res = self.hamamatsu_cam.set_binning(binning_val)
            if isinstance(res, tuple):
                self.width, self.height = res
                if dpg.does_item_exist(self.width_tag):
                    dpg.set_value(self.width_tag, self.width)
                if dpg.does_item_exist(self.height_tag):
                    dpg.set_value(self.height_tag, self.height)

            if was_running:
                self._start_capture()

    def _apply_aoi(self, sender: Optional[str] = None, app_data: Any = None, user_data: Any = None, *args: Any) -> None:
        """
        Apply Area of Interest (AOI) / subarray resolution changes.

        Dimensions are clamped to even numbers as required by camera drivers.
        Image buffers are reallocated and any existing dark reference frame is cleared.
        """
        if self.hamamatsu_cam.defined:
            was_running = self.cam_running
            if was_running:
                self._stop_capture()

            self.dark_ref_frame = None
            w = dpg.get_value(self.width_tag)
            h = dpg.get_value(self.height_tag)

            w = (w // 2) * 2
            h = (h // 2) * 2

            res = self.hamamatsu_cam.set_aoi(w, h)
            if isinstance(res, tuple):
                self.width, self.height = res
                dpg.set_value(self.width_tag, self.width)
                dpg.set_value(self.height_tag, self.height)

            if was_running:
                self._start_capture()

            logger.info(f"Hamamatsu Resolution updated to {self.width}x{self.height}")

    def _start_dark_ref_capture(self, sender: Optional[str] = None, app_data: Any = None, user_data: Any = None, *args: Any) -> None:
        """
        Start an automated sequence to capture and average dark reference frames.
        Temporarily switches to Trigger Software mode and suppresses regular output emissions.
        """
        self.dark_ref_accumulator = None
        self.dark_ref_counter = 0
        self.collecting_dark_ref = True
        self._suppress_emission = True
        logger.info(f"Starting dark reference capture ({self.dark_ref_n_images} frames)")

        original_mode = dpg.get_value(self.trigger_tag)

        if original_mode != "Trigger Software":
            self._set_trigger(None, "Trigger Software")
            dpg.set_value(self.trigger_tag, "Trigger Software")

        def capture_sequence() -> None:
            time.sleep(0.1)
            expos_us = dpg.get_value(self.exposure_tag)
            sleep_time = (expos_us / 1000000.0) + 0.1

            for _ in range(self.dark_ref_n_images):
                if not self.collecting_dark_ref:
                    break
                self.hamamatsu_cam.trigger()
                time.sleep(sleep_time)

            timeout = 2.0
            start = time.time()
            while self.collecting_dark_ref and time.time() - start < timeout:
                time.sleep(0.1)

            self.collecting_dark_ref = False

            if original_mode != "Trigger Software":
                self._set_trigger(None, original_mode)
                dpg.set_value(self.trigger_tag, original_mode)

            time.sleep(0.5)
            self._suppress_emission = False

        threading.Thread(target=capture_sequence, daemon=True).start()

    def _clear_dark_ref(self, sender: Optional[str] = None, app_data: Any = None, user_data: Any = None, *args: Any) -> None:
        """
        Clear the stored dark reference calibration frame.
        """
        self.dark_ref_frame = None
        self.collecting_dark_ref = False
        logger.info("Dark reference cleared")

    def is_ready(self) -> bool:
        """
        Check if the camera module is ready to process input commands.

        Returns:
            bool: Always True for the camera window.
        """
        return True

    def input_cb(self, data: Any = None, *args: Any, **kwargs: Any) -> None:
        """
        Handle incoming command dictionary from connected modules or scripts.

        Supported command keys:
            - init (bool): Initialize the camera hardware.
            - start (bool): Start continuous frame acquisition.
            - stop (bool): Stop continuous acquisition.
            - trigger (bool): Software trigger a single frame.
            - trigger_cam (bool/dict): Trigger camera, optionally with a temporary exposure time.
            - exposure (int): Set exposure time in microseconds.
            - binning (int): Set binning factor (1, 2, or 4).
            - readout_speed (str): Set readout speed ("Fastest" or "Slowest").
            - trigger_mode (str): Set trigger mode ("Video", "Trigger", or "Trigger Software").
            - start_dark_ref (bool): Start dark reference acquisition.
            - n_frames (int): Set dark reference repetition count.
            - width (int) & height (int): Set AOI resolution.
        """
        if data is None and "data" in kwargs:
            data = kwargs["data"]

        if isinstance(data, dict):
            cmd = data
        elif kwargs.get("data_type") == IOTypes.CMD_DICT:
            cmd = kwargs.get("data")
        else:
            cmd = None

        if cmd and isinstance(cmd, dict):
            if "init" in cmd and cmd["init"]:
                self._init_camera()

            if "start" in cmd and cmd["start"]:
                self._start_capture()

            if "stop" in cmd and cmd["stop"]:
                self._stop_capture()

            if "trigger_cam" in cmd:
                if "exposure" in cmd:
                    def trigger_seq() -> None:
                        old_exposure = dpg.get_value(self.exposure_tag)
                        new_exposure = int(cmd["exposure"])
                        self._set_exposure(None, new_exposure)
                        dpg.set_value(self.exposure_tag, new_exposure)
                        time.sleep(0.1)
                        self.hamamatsu_cam.trigger()
                        time.sleep((new_exposure / 1000000.0) + 0.1)
                        self._set_exposure(None, old_exposure)
                        dpg.set_value(self.exposure_tag, old_exposure)

                    threading.Thread(target=trigger_seq, daemon=True).start()
                else:
                    self.hamamatsu_cam.trigger()

            if "trigger" in cmd:
                self.hamamatsu_cam.trigger()

            if "exposure" in cmd and "trigger_cam" not in cmd and "start_dark_ref" not in cmd:
                val = int(cmd["exposure"])
                self._set_exposure(None, val)
                dpg.set_value(self.exposure_tag, val)

            if "start_dark_ref" in cmd:
                if "n_frames" in cmd:
                    self.dark_ref_n_images = int(cmd["n_frames"])
                self._start_dark_ref_capture()

            if "trigger_mode" in cmd:
                mode = cmd["trigger_mode"]
                if mode in ["Trigger", "hardware", "external"]:
                    mode = "Trigger"
                elif mode in ["Trigger Software", "software", "Trigger_Software", "TriggerSoft"]:
                    mode = "Trigger Software"
                elif mode in ["Video", "internal"]:
                    mode = "Video"
                self._set_trigger(None, mode)
                dpg.set_value(self.trigger_tag, mode)

            if "binning" in cmd:
                val = int(cmd["binning"])
                if val in [1, 2, 4]:
                    self._set_binning(None, f"{val}x{val}")
                    dpg.set_value(self.binning_tag, f"{val}x{val}")

            if "readout_speed" in cmd:
                val = cmd["readout_speed"]
                if val in ["Fastest", "Slowest"]:
                    self._set_readout_speed(None, val)
                    dpg.set_value(self.readout_speed_tag, val)

            if "width" in cmd and "height" in cmd:
                self.width = int(cmd["width"])
                self.height = int(cmd["height"])
                dpg.set_value(self.width_tag, self.width)
                dpg.set_value(self.height_tag, self.height)
                self._apply_aoi()

    def close(self) -> None:
        """
        Stop acquisition and release camera hardware resources upon window closing.
        """
        self._stop_capture()
        if self.hamamatsu_cam.defined:
            try:
                self.hamamatsu_cam.close()
            except Exception:
                pass
        super().close()


EXPORTED_CLASS = Hamamatsu_win
EXPORTED_NAME = "Hamamatsu Camera"
