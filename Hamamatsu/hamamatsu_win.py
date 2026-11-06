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
from .hamamatsu import Hamamatsu

_ROTATION_MAP = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


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

    label: str = "Hamamatsu Camera"
    win_width: int = display_scaling.scale(350)
    win_height: int = display_scaling.scale(300)
    description: str = (
        "Control Hamamatsu scientific cameras. Supports exposure, trigger modes, software trigger, and 16-bit frame output."
    )

    accepted_input_types: list[str] = [IOTypes.CMD_DICT]
    outputs: dict[str, str] = {
        "16bit frame": IOTypes.FRAME16,
    }
    persistent_fields: list[str] = ["label", "width", "height", "binning", "readout_speed"]

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the Hamamatsu Camera window module.

        Args:
            **kwargs: Additional keyword arguments forwarded to WindowBase.
        """
        super().__init__(**kwargs)

        uid = self.UUID
        self.init_btn_tag: str = f"hama_init_btn_{uid}"
        self.start_btn_tag: str = f"hama_start_btn_{uid}"
        self.exposure_tag: str = f"hama_exposure_{uid}"
        self.trigger_tag: str = f"hama_trigger_{uid}"
        self.width_tag: str = f"hama_width_{uid}"
        self.height_tag: str = f"hama_height_{uid}"
        self.binning_tag: str = f"hama_binning_{uid}"
        self.readout_speed_tag: str = f"hama_readout_speed_{uid}"

        self.width: Optional[int] = getattr(self, "width", None)
        self.height: Optional[int] = getattr(self, "height", None)
        self.binning: int = int(getattr(self, "binning", None) or 1)
        self.readout_speed: str = str(getattr(self, "readout_speed", None) or "Fastest")

        self._suppress_emission: bool = False
        self.cam_running: bool = False
        self.cam_thread: Optional[threading.Thread] = None
        self.hamamatsu_cam: Hamamatsu = Hamamatsu()

        self.dark_ref_frame: Optional[np.ndarray] = None
        self.collecting_dark_ref: bool = False
        self.dark_ref_n_images: int = 10
        self.dark_ref_accumulator: Optional[np.ndarray] = None
        self.dark_ref_counter: int = 0
        self.correct_dark: bool = True
        self.dark_threshold: int = 10

        self._theme_init: Optional[int] = None
        self._theme_success: Optional[int] = None
        self._theme_fail: Optional[int] = None

        angle = config.get("Cameras", {}).get("Hamamatsu", {}).get("rotation", 0)
        self._rotation_code: Optional[int] = _ROTATION_MAP.get(angle)

        if app_state.get("detector") is None:
            app_state.detector = 50000

        self._build_ui()
        self.autosize_window()
        self.update_permission()

    def _ensure_themes(self) -> None:
        """Lazily create reusable button status themes to avoid DPG item leaks."""
        if self._theme_init is None or not dpg.does_item_exist(self._theme_init):
            with dpg.theme() as self._theme_init:
                with dpg.theme_component(dpg.mvAll):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (255, 165, 0), category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (255, 180, 50), category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (255, 140, 0), category=dpg.mvThemeCat_Core)

        if self._theme_success is None or not dpg.does_item_exist(self._theme_success):
            with dpg.theme() as self._theme_success:
                with dpg.theme_component(dpg.mvAll):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (0, 100, 0), category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (0, 150, 0), category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (0, 200, 0), category=dpg.mvThemeCat_Core)

        if self._theme_fail is None or not dpg.does_item_exist(self._theme_fail):
            with dpg.theme() as self._theme_fail:
                with dpg.theme_component(dpg.mvAll):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (150, 0, 0), category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (200, 0, 0), category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (255, 0, 0), category=dpg.mvThemeCat_Core)

    def _build_ui(self) -> None:
        """Construct DearPyGui widgets for Hamamatsu camera control."""
        scale = display_scaling.scale

        with self.window():
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
                    width=scale(150),
                )
                dpg.add_button(
                    label="Trigger",
                    callback=lambda: self.hamamatsu_cam.trigger(),
                )

            with dpg.group(horizontal=True):
                dpg.add_drag_int(
                    label="Expos. (µs)",
                    default_value=50000,
                    min_value=1000,
                    max_value=10000000,
                    tag=self.exposure_tag,
                    callback=self._set_exposure,
                    width=scale(100),
                )
                dpg.add_combo(
                    items=["1x1", "2x2", "4x4"],
                    default_value=f"{self.binning}x{self.binning}",
                    tag=self.binning_tag,
                    callback=self._set_binning,
                    width=scale(80),
                )
                dpg.add_combo(
                    items=["Slowest", "Fastest"],
                    default_value=self.readout_speed,
                    tag=self.readout_speed_tag,
                    callback=self._set_readout_speed,
                    width=scale(80),
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
                    width=scale(100),
                    step=-1,
                    callback=lambda s, a: setattr(self, "dark_ref_n_images", a),
                )

                dpg.add_checkbox(
                    label="Correct dark",
                    default_value=self.correct_dark,
                    callback=lambda s, a: setattr(self, "correct_dark", a),
                )

                dpg.add_input_int(
                    label="Dark threshold",
                    default_value=self.dark_threshold,
                    width=scale(100),
                    step=-1,
                    callback=lambda s, a: setattr(self, "dark_threshold", a),
                )

                dpg.add_separator()
                dpg.add_text("AOI / Resolution")
                with dpg.group(horizontal=True):
                    dpg.add_input_int(
                        label="Width",
                        default_value=self.width if self.width is not None else 2048,
                        tag=self.width_tag,
                        width=scale(100),
                        step=0,
                        callback=lambda s, a: setattr(self, "width", a),
                    )
                    dpg.add_input_int(
                        label="Height",
                        default_value=self.height if self.height is not None else 2048,
                        tag=self.height_tag,
                        width=scale(100),
                        step=0,
                        callback=lambda s, a: setattr(self, "height", a),
                    )
                dpg.add_button(
                    label="Update Resolution", callback=self._apply_aoi
                )

    def update_permission(self) -> None:
        """
        Adjust module permissions and UI visibility based on the application mode.
        """
        is_user = app_state.mode == "user"
        if dpg.does_item_exist(self.winID):
            dpg.configure_item(self.winID, no_close=is_user)

    def _init_camera(self, *args: Any, **kwargs: Any) -> None:
        """
        Asynchronously initialize the Hamamatsu camera hardware, configure
        initial exposure, trigger, binning, and readout parameters, and start continuous capture.
        """
        self._ensure_themes()
        if self._theme_init and dpg.does_item_exist(self.init_btn_tag):
            dpg.bind_item_theme(self.init_btn_tag, self._theme_init)

        def init_task() -> None:
            try:
                readout_speed_val = (
                    dpg.get_value(self.readout_speed_tag)
                    if dpg.does_item_exist(self.readout_speed_tag)
                    else self.readout_speed
                )
                self.hamamatsu_cam.readout_speed = readout_speed_val

                exposure_us = (
                    dpg.get_value(self.exposure_tag)
                    if dpg.does_item_exist(self.exposure_tag)
                    else 50000
                )
                trigger_mode = (
                    dpg.get_value(self.trigger_tag)
                    if dpg.does_item_exist(self.trigger_tag)
                    else "Trigger"
                )
                binning_str = (
                    dpg.get_value(self.binning_tag)
                    if dpg.does_item_exist(self.binning_tag)
                    else f"{self.binning}x{self.binning}"
                )
                binning_val = int(str(binning_str).split("x")[0])

                if self.hamamatsu_cam.init(
                    exposure_us=float(exposure_us),
                    trigger_mode=str(trigger_mode),
                    binning_factor=binning_val,
                ):
                    if self._theme_success and dpg.does_item_exist(self.init_btn_tag):
                        dpg.bind_item_theme(self.init_btn_tag, self._theme_success)

                    logger.success("Hamamatsu camera initialized")

                    try:
                        app_state.detector = int(exposure_us)
                    except (ValueError, TypeError):
                        pass

                    self.width = int(self.hamamatsu_cam.width)
                    self.height = int(self.hamamatsu_cam.height)

                    if dpg.does_item_exist(self.width_tag):
                        dpg.set_value(self.width_tag, self.width)
                    if dpg.does_item_exist(self.height_tag):
                        dpg.set_value(self.height_tag, self.height)

                    self._start_capture()
                else:
                    if self._theme_fail and dpg.does_item_exist(self.init_btn_tag):
                        dpg.bind_item_theme(self.init_btn_tag, self._theme_fail)
                    logger.error("Hamamatsu camera initialization failed")
            except Exception as e:
                if self._theme_fail and dpg.does_item_exist(self.init_btn_tag):
                    dpg.bind_item_theme(self.init_btn_tag, self._theme_fail)
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
        if self._rotation_code is not None:
            return cv2.rotate(frame, self._rotation_code)
        return frame

    def _start_capture(self, *args: Any, **kwargs: Any) -> None:
        """
        Start the background acquisition thread to capture frames from the camera.
        """
        if not self.hamamatsu_cam.defined:
            return

        if self.cam_thread and self.cam_thread.is_alive():
            return

        self.cam_running = True
        self._ensure_themes()
        if self._theme_success and dpg.does_item_exist(self.start_btn_tag):
            dpg.bind_item_theme(self.start_btn_tag, self._theme_success)

        def capture_task() -> None:
            while self.cam_running:
                if not self.hamamatsu_cam.defined or self.hamamatsu_cam.hwait is None:
                    time.sleep(0.05)
                    continue

                if self.hamamatsu_cam.check_event(timeout_ms=100):
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
                            corrected = frame.astype(np.float32)
                            np.subtract(corrected, ref_frame, out=corrected)

                            masked_vals = corrected[corrected < self.dark_threshold]
                            if masked_vals.size > 0:
                                corrected -= float(masked_vals.mean())

                            np.clip(corrected, 0.0, 65535.0, out=corrected)
                            frame = corrected.astype(np.uint16)

                        if not is_collecting and not self._suppress_emission:
                            self._trigger_outputs(frame)

        self.cam_thread = threading.Thread(target=capture_task, daemon=True)
        self.cam_thread.start()

    def _stop_capture(self, *args: Any, **kwargs: Any) -> None:
        """
        Stop the background acquisition thread and reset Start button styling.
        """
        if not self.cam_running:
            return
        self.cam_running = False
        if self.cam_thread:
            self.cam_thread.join(timeout=1.0)
        if dpg.does_item_exist(self.start_btn_tag):
            dpg.bind_item_theme(self.start_btn_tag, 0)

    def _trigger_outputs(self, frame: np.ndarray) -> None:
        """
        Dispatch the captured frame tuple (frame, "Hamamatsu") to all connected modules.

        Args:
            frame: 16-bit monochrome frame array (uint16).
        """
        data = (frame, "Hamamatsu")
        for _output_key, connected_modules in self.connections.items():
            for module in connected_modules:
                try:
                    module.input_cb(data)
                except Exception as e:
                    logger.error(f"Error sending frame to {module}: {e}")

    def _set_exposure(self, sender: Any = None, app_data: Any = None, *args: Any, **kwargs: Any) -> None:
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

    def _set_trigger(self, sender: Any = None, app_data: Any = None, *args: Any, **kwargs: Any) -> None:
        """
        Set acquisition trigger mode ("Video", "Trigger", or "Trigger Software").
        """
        if self.hamamatsu_cam.defined and app_data is not None:
            self.hamamatsu_cam.set_trigger(app_data)

    def _set_readout_speed(self, sender: Any = None, app_data: Any = None, *args: Any, **kwargs: Any) -> None:
        """
        Set sensor readout speed ("Fastest" or "Slowest").
        """
        if app_data is None:
            return
        self.readout_speed = str(app_data)
        self.dark_ref_frame = None
        if self.hamamatsu_cam.defined:
            was_running = self.cam_running
            if was_running:
                self._stop_capture()
            self.hamamatsu_cam.set_readout_speed(self.readout_speed)
            if was_running:
                self._start_capture()

    def _set_binning(self, sender: Any = None, app_data: Any = None, *args: Any, **kwargs: Any) -> None:
        """
        Set sensor binning factor (1x1, 2x2, 4x4) and update resolution fields.
        """
        if app_data is None:
            return
        binning_val = int(str(app_data).split("x")[0])
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

    def _apply_aoi(self, *args: Any, **kwargs: Any) -> None:
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
                if dpg.does_item_exist(self.width_tag):
                    dpg.set_value(self.width_tag, self.width)
                if dpg.does_item_exist(self.height_tag):
                    dpg.set_value(self.height_tag, self.height)

            if was_running:
                self._start_capture()

            logger.info(f"Hamamatsu Resolution updated to {self.width}x{self.height}")

    def _start_dark_ref_capture(self, *args: Any, **kwargs: Any) -> None:
        """
        Start an automated sequence to capture and average dark reference frames.
        Temporarily switches to Trigger Software mode and suppresses regular output emissions.
        """
        self.dark_ref_accumulator = None
        self.dark_ref_counter = 0
        self.collecting_dark_ref = True
        self._suppress_emission = True
        logger.info(f"Starting dark reference capture ({self.dark_ref_n_images} frames)")

        original_mode = (
            dpg.get_value(self.trigger_tag)
            if dpg.does_item_exist(self.trigger_tag)
            else "Trigger Software"
        )

        if original_mode != "Trigger Software":
            self._set_trigger(None, "Trigger Software")
            if dpg.does_item_exist(self.trigger_tag):
                dpg.set_value(self.trigger_tag, "Trigger Software")

        def capture_sequence() -> None:
            time.sleep(0.1)
            expos_us = (
                dpg.get_value(self.exposure_tag)
                if dpg.does_item_exist(self.exposure_tag)
                else 50000
            )
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
                if dpg.does_item_exist(self.trigger_tag):
                    dpg.set_value(self.trigger_tag, original_mode)

            time.sleep(0.5)
            self._suppress_emission = False

        threading.Thread(target=capture_sequence, daemon=True).start()

    def _clear_dark_ref(self, *args: Any, **kwargs: Any) -> None:
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

        if not cmd or not isinstance(cmd, dict):
            return

        if "init" in cmd and cmd["init"]:
            self._init_camera()

        if "start" in cmd and cmd["start"]:
            self._start_capture()

        if "stop" in cmd and cmd["stop"]:
            self._stop_capture()

        if "trigger_cam" in cmd:
            if "exposure" in cmd:
                def trigger_seq() -> None:
                    old_exposure = (
                        dpg.get_value(self.exposure_tag)
                        if dpg.does_item_exist(self.exposure_tag)
                        else 50000
                    )
                    new_exposure = int(cmd["exposure"])
                    self._set_exposure(None, new_exposure)
                    if dpg.does_item_exist(self.exposure_tag):
                        dpg.set_value(self.exposure_tag, new_exposure)
                    time.sleep(0.1)
                    self.hamamatsu_cam.trigger()
                    time.sleep((new_exposure / 1000000.0) + 0.1)
                    self._set_exposure(None, old_exposure)
                    if dpg.does_item_exist(self.exposure_tag):
                        dpg.set_value(self.exposure_tag, old_exposure)

                threading.Thread(target=trigger_seq, daemon=True).start()
            else:
                self.hamamatsu_cam.trigger()

        if "trigger" in cmd:
            self.hamamatsu_cam.trigger()

        if "exposure" in cmd and "trigger_cam" not in cmd and "start_dark_ref" not in cmd:
            val = int(cmd["exposure"])
            self._set_exposure(None, val)
            if dpg.does_item_exist(self.exposure_tag):
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
            if dpg.does_item_exist(self.trigger_tag):
                dpg.set_value(self.trigger_tag, mode)

        if "binning" in cmd:
            val = int(cmd["binning"])
            if val in [1, 2, 4]:
                self._set_binning(None, f"{val}x{val}")
                if dpg.does_item_exist(self.binning_tag):
                    dpg.set_value(self.binning_tag, f"{val}x{val}")

        if "readout_speed" in cmd:
            speed_val = str(cmd["readout_speed"])
            if speed_val in ["Fastest", "Slowest"]:
                self._set_readout_speed(None, speed_val)
                if dpg.does_item_exist(self.readout_speed_tag):
                    dpg.set_value(self.readout_speed_tag, speed_val)

        if "width" in cmd and "height" in cmd:
            self.width = int(cmd["width"])
            self.height = int(cmd["height"])
            if dpg.does_item_exist(self.width_tag):
                dpg.set_value(self.width_tag, self.width)
            if dpg.does_item_exist(self.height_tag):
                dpg.set_value(self.height_tag, self.height)
            self._apply_aoi()

    def on_close(self) -> None:
        """
        Stop acquisition and release camera hardware resources upon window closure.
        """
        self._stop_capture()
        if hasattr(self, "hamamatsu_cam") and getattr(self.hamamatsu_cam, "defined", False):
            try:
                self.hamamatsu_cam.close()
            except Exception:
                pass
        super().on_close()

    def close(self) -> None:
        """
        Stop acquisition and release camera hardware resources upon window closing.
        """
        self.on_close()
        super().close()


EXPORTED_CLASS = Hamamatsu_win
EXPORTED_NAME = "Hamamatsu Camera"
