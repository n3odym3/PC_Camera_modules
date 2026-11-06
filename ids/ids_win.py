from __future__ import annotations

import os
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
from .ids import IDS

_MODULE_DIR = os.path.dirname(__file__)
_FFC_MONO_PATH = os.path.join(_MODULE_DIR, "ids_ffc_mono.npy")
_FFC_RGB_PATH = os.path.join(_MODULE_DIR, "ids_ffc_rgb.npy")
_ROTATION_MAP = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


class IDS_win(WindowBase):
    """
    Window module for controlling IDS Imaging Development Systems industrial cameras.

    Features:
        - Continuous live acquisition ("Video") and hardware/software ("Trigger") modes.
        - 12-bit monochrome frame streaming (output as 16-bit container `uint16`).
        - Real-time dark reference accumulation and background subtraction with dynamic offset.
        - Flat-Field Correction (FFC) with separate Mono and RGB calibration matrices.
        - Area of Interest (AOI) / resolution configuration with automatic memory reallocation.
        - Remote automation control via dictionary commands (`IOTypes.CMD_DICT`).
    """

    label: str = "IDS Camera"
    win_width: int = display_scaling.scale(350)
    win_height: int = display_scaling.scale(300)
    description: str = (
        "Control IDS industrial cameras. Supports exposure, gain, trigger modes, FFC, and 12-bit frame output."
    )

    accepted_input_types: list[str] = [IOTypes.CMD_DICT]
    outputs: dict[str, str] = {
        "12bit frame": IOTypes.FRAME12,
    }
    persistent_fields: list[str] = ["label", "width", "height", "ffc_enabled"]

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the IDS Camera window module.

        Args:
            **kwargs: Additional keyword arguments forwarded to WindowBase.
        """
        super().__init__(**kwargs)

        uid = self.UUID
        self.init_btn_tag: str = f"ids_init_btn_{uid}"
        self.start_btn_tag: str = f"ids_start_btn_{uid}"
        self.trigger_tag: str = f"ids_trigger_{uid}"
        self.exposure_tag: str = f"ids_exposure_{uid}"
        self.gain_tag: str = f"ids_gain_{uid}"
        self.width_tag: str = f"ids_width_{uid}"
        self.height_tag: str = f"ids_height_{uid}"
        self.ffc_enabled_tag: str = f"ids_ffc_enabled_{uid}"
        self.ffc_status_tag: str = f"ids_ffc_status_{uid}"

        self.width: Optional[int] = getattr(self, "width", None)
        self.height: Optional[int] = getattr(self, "height", None)
        self.ffc_enabled: bool = bool(getattr(self, "ffc_enabled", False))

        self._suppress_emission: bool = False
        self.cam_running: bool = False
        self.cam_thread: Optional[threading.Thread] = None
        self.ids_cam: IDS = IDS()

        self.dark_ref_frame: Optional[np.ndarray] = None
        self.collecting_dark_ref: bool = False
        self.dark_ref_n_images: int = 10
        self.dark_ref_accumulator: Optional[np.ndarray] = None
        self.dark_ref_counter: int = 0
        self.correct_dark: bool = True
        self.dark_threshold: int = 10

        self.ffc_matrix_mono: Optional[np.ndarray] = None
        self.ffc_matrix_rgb: Optional[np.ndarray] = None
        self.ffc_gain_mono: Optional[np.ndarray] = None
        self.ffc_gain_rgb: Optional[np.ndarray] = None
        self._ffc_gain_mono_cached: Optional[np.ndarray] = None
        self._ffc_gain_rgb_cached: Optional[np.ndarray] = None
        self._capture_ffc_next: bool = False

        self._theme_init: Optional[int] = None
        self._theme_success: Optional[int] = None
        self._theme_fail: Optional[int] = None

        angle = config.get("Cameras", {}).get("IDS", {}).get("rotation", 0)
        self._rotation_code: Optional[int] = _ROTATION_MAP.get(angle)

        if app_state.get("detector") is None:
            app_state.detector = 2000

        self._build_ui()
        self._load_ffc_matrices()
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
        """Construct DearPyGui widgets for IDS camera control."""
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
                    items=["Video", "Trigger"],
                    default_value="Video",
                    tag=self.trigger_tag,
                    callback=self._set_trigger,
                    width=scale(150),
                )
                dpg.add_button(
                    label="Trigger",
                    callback=lambda: self.ids_cam.trigger(),
                )

            dpg.add_drag_int(
                label="Expos. (µs)",
                default_value=2000,
                min_value=1,
                max_value=5000,
                tag=self.exposure_tag,
                callback=self._set_exposure,
                width=scale(100),
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

                dpg.add_slider_int(
                    label="Gain",
                    default_value=0,
                    min_value=0,
                    max_value=100,
                    tag=self.gain_tag,
                    callback=self._set_gain,
                    width=scale(200),
                )

                dpg.add_separator()
                dpg.add_text("Flat Field Correction (FFC)")
                dpg.add_checkbox(
                    label="Enable FFC",
                    tag=self.ffc_enabled_tag,
                    default_value=self.ffc_enabled,
                    callback=lambda s, v: setattr(self, "ffc_enabled", v),
                )
                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="Take Ref Image",
                        callback=self._request_ffc_capture,
                    )
                    dpg.add_text(
                        "None",
                        tag=self.ffc_status_tag,
                        color=(200, 100, 100),
                    )

                dpg.add_separator()
                dpg.add_text("AOI / Resolution")
                with dpg.group(horizontal=True):
                    dpg.add_input_int(
                        label="Width",
                        default_value=self.width if self.width is not None else 1280,
                        tag=self.width_tag,
                        width=scale(100),
                        step=0,
                        callback=lambda s, a: setattr(self, "width", a),
                    )
                    dpg.add_input_int(
                        label="Height",
                        default_value=self.height if self.height is not None else 1024,
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
        Asynchronously initialize the IDS camera hardware, retrieve sensor geometry,
        and start continuous frame capture.
        """
        self._ensure_themes()
        if self._theme_init and dpg.does_item_exist(self.init_btn_tag):
            dpg.bind_item_theme(self.init_btn_tag, self._theme_init)

        def init_task() -> None:
            try:
                if self.ids_cam.init():
                    if self._theme_success and dpg.does_item_exist(self.init_btn_tag):
                        dpg.bind_item_theme(self.init_btn_tag, self._theme_success)

                    logger.success("IDS camera initialized")

                    self.width = int(self.ids_cam.width)
                    self.height = int(self.ids_cam.height)

                    if dpg.does_item_exist(self.width_tag):
                        dpg.set_value(self.width_tag, self.width)
                    if dpg.does_item_exist(self.height_tag):
                        dpg.set_value(self.height_tag, self.height)

                    self._start_capture()
                else:
                    if self._theme_fail and dpg.does_item_exist(self.init_btn_tag):
                        dpg.bind_item_theme(self.init_btn_tag, self._theme_fail)
                    logger.error("IDS camera initialization failed")
            except Exception as e:
                if self._theme_fail and dpg.does_item_exist(self.init_btn_tag):
                    dpg.bind_item_theme(self.init_btn_tag, self._theme_fail)
                logger.error(f"IDS init error: {e}")

        threading.Thread(target=init_task, daemon=True).start()

    def _apply_rotation(self, frame: np.ndarray) -> np.ndarray:
        """
        Rotate frame based on camera configuration (Cameras.IDS.rotation).

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
        if not self.ids_cam.defined:
            return

        if self.cam_thread and self.cam_thread.is_alive():
            return

        self.cam_running = True
        self._ensure_themes()
        if self._theme_success and dpg.does_item_exist(self.start_btn_tag):
            dpg.bind_item_theme(self.start_btn_tag, self._theme_success)

        def capture_task() -> None:
            while self.cam_running:
                if self.ids_cam.check_event(timeout_ms=100):
                    frame = self.ids_cam.get_image()
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
        Process the acquired frame (FFC) and dispatch to outputs.

        Args:
            frame: 16-bit container frame (uint16).
        """
        if self._capture_ffc_next:
            self._capture_ffc_next = False
            self._process_ffc_reference(frame)

        if self.ffc_enabled:
            try:
                is_color = frame.ndim == 3
                base_gain = self.ffc_gain_rgb if is_color else self.ffc_gain_mono

                if base_gain is not None:
                    dtype = frame.dtype
                    h, w = frame.shape[:2]

                    if is_color:
                        cached = self._ffc_gain_rgb_cached
                        if cached is None or cached.shape[:2] != (h, w):
                            resized = cv2.resize(base_gain, (w, h)) if base_gain.shape[:2] != (h, w) else base_gain
                            cached = np.expand_dims(resized, axis=2)
                            self._ffc_gain_rgb_cached = cached
                    else:
                        cached = self._ffc_gain_mono_cached
                        if cached is None or cached.shape[:2] != (h, w):
                            cached = cv2.resize(base_gain, (w, h)) if base_gain.shape[:2] != (h, w) else base_gain
                            self._ffc_gain_mono_cached = cached

                    frame_f = frame.astype(np.float32)
                    np.multiply(frame_f, cached, out=frame_f)

                    is_12bit = True
                    max_clip = 4095.0 if is_12bit else (65535.0 if dtype == np.uint16 else 255.0)
                    np.clip(frame_f, 0.0, max_clip, out=frame_f)
                    frame = frame_f.astype(dtype)
            except Exception as e:
                logger.error(f"FFC application failed: {e}")

        data = (frame, "IDS")
        for _output_key, connected_modules in self.connections.items():
            for module in connected_modules:
                try:
                    module.input_cb(data)
                except Exception as e:
                    logger.error(f"Error sending frame to {module}: {e}")

    def _request_ffc_capture(self, *args: Any, **kwargs: Any) -> None:
        """
        Flag the next incoming frame for Flat-Field Correction (FFC) reference calculation.
        """
        if not self.cam_running:
            logger.warning("Camera must be running to take FFC reference")
            return
        self._capture_ffc_next = True
        logger.info("FFC reference capture requested - point camera at uniform source")

    def _process_ffc_reference(self, frame: np.ndarray) -> None:
        """
        Calculate and persist flat-field normalization matrix from a uniform illumination frame.

        Args:
            frame: Input image of uniform background.
        """
        try:
            is_color = len(frame.shape) == 3

            if is_color:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame.copy()

            gray_f = gray.astype(np.float32)
            gray_f = cv2.GaussianBlur(gray_f, (31, 31), 0)

            peak = np.max(gray_f)
            if peak == 0:
                peak = 1.0

            matrix = gray_f / peak
            matrix[matrix < 0.05] = 0.05
            gain = 1.0 / matrix

            if is_color:
                self.ffc_matrix_rgb = matrix
                self.ffc_gain_rgb = gain
                self._ffc_gain_rgb_cached = np.expand_dims(gain, axis=2)
                f_path = _FFC_RGB_PATH
            else:
                self.ffc_matrix_mono = matrix
                self.ffc_gain_mono = gain
                self._ffc_gain_mono_cached = gain
                f_path = _FFC_MONO_PATH

            np.save(f_path, matrix)

            if dpg.does_item_exist(self.ffc_status_tag):
                dpg.set_value(self.ffc_status_tag, f"OK ({'RGB' if is_color else 'Mono'})")
                dpg.configure_item(self.ffc_status_tag, color=(100, 255, 100))
            logger.success(f"FFC {'RGB' if is_color else 'Mono'} reference captured as Grayscale matrix")

        except Exception as e:
            logger.error(f"FFC processing failed: {e}")
            if dpg.does_item_exist(self.ffc_status_tag):
                dpg.set_value(self.ffc_status_tag, "Error")
                dpg.configure_item(self.ffc_status_tag, color=(255, 100, 100))

    def _load_ffc_matrices(self) -> None:
        """
        Load persisted Flat-Field Correction matrices from disk if available.
        """
        status_parts: list[str] = []
        if os.path.exists(_FFC_MONO_PATH):
            try:
                self.ffc_matrix_mono = np.load(_FFC_MONO_PATH)
                self.ffc_gain_mono = 1.0 / self.ffc_matrix_mono
                self._ffc_gain_mono_cached = self.ffc_gain_mono
                status_parts.append("Mono")
            except Exception as e:
                logger.warning(f"Failed to load IDS FFC mono matrix: {e}")
        if os.path.exists(_FFC_RGB_PATH):
            try:
                self.ffc_matrix_rgb = np.load(_FFC_RGB_PATH)
                self.ffc_gain_rgb = 1.0 / self.ffc_matrix_rgb
                self._ffc_gain_rgb_cached = np.expand_dims(self.ffc_gain_rgb, axis=2)
                status_parts.append("RGB")
            except Exception as e:
                logger.warning(f"Failed to load IDS FFC rgb matrix: {e}")

        if status_parts and dpg.does_item_exist(self.ffc_status_tag):
            dpg.set_value(self.ffc_status_tag, f"OK ({'+'.join(status_parts)})")
            dpg.configure_item(self.ffc_status_tag, color=(100, 255, 100))

    def _set_pixel_clock(self, sender: Any = None, app_data: Any = None, *args: Any, **kwargs: Any) -> None:
        """
        Set the camera pixel clock in MHz.
        """
        if self.ids_cam.defined and app_data is not None:
            self.ids_cam.set_pixel_clock(app_data)

    def _set_gain(self, sender: Any = None, app_data: Any = None, *args: Any, **kwargs: Any) -> None:
        """
        Set camera hardware gain (0-100).
        """
        if self.ids_cam.defined and app_data is not None:
            self.ids_cam.set_gain(app_data)

    def _set_exposure(self, sender: Any = None, app_data: Any = None, *args: Any, **kwargs: Any) -> None:
        """
        Set camera exposure time in microseconds and update global detector state.
        """
        if app_data is not None:
            try:
                app_state.detector = int(app_data)
            except (ValueError, TypeError):
                pass
            if self.ids_cam.defined:
                self.ids_cam.set_exposure(app_data)

    def _set_shutter(self, sender: Any = None, app_data: Any = None, *args: Any, **kwargs: Any) -> None:
        """
        Set sensor shutter mode ("Global", "Global alt.", "Rolling", "Rolling glob.").
        """
        if self.ids_cam.defined and app_data is not None:
            self.ids_cam.set_shutter(app_data)

    def _set_trigger(self, sender: Any = None, app_data: Any = None, *args: Any, **kwargs: Any) -> None:
        """
        Set acquisition trigger mode ("Video" for continuous stream, "Trigger" for external/software trigger).
        """
        if self.ids_cam.defined and app_data is not None:
            self.ids_cam.set_trigger(app_data)

    def _apply_aoi(self, *args: Any, **kwargs: Any) -> None:
        """
        Apply Area of Interest (AOI) / resolution changes.

        Dimensions are clamped to even numbers as required by the sensor driver.
        Image buffers are reallocated and any existing dark reference frame is cleared.
        """
        if self.ids_cam.defined:
            was_running = self.cam_running
            if was_running:
                self._stop_capture()

            w = dpg.get_value(self.width_tag)
            h = dpg.get_value(self.height_tag)

            # Ensure even dimensions as required by uEye sensors
            w = (w // 2) * 2
            h = (h // 2) * 2

            res = self.ids_cam.set_aoi(w, h)
            if isinstance(res, tuple):
                self.width, self.height = res
                if dpg.does_item_exist(self.width_tag):
                    dpg.set_value(self.width_tag, self.width)
                if dpg.does_item_exist(self.height_tag):
                    dpg.set_value(self.height_tag, self.height)

            # Reset dark reference and cached FFC since dimensions changed
            self.dark_ref_frame = None
            self._ffc_gain_mono_cached = None
            self._ffc_gain_rgb_cached = None

            if was_running:
                self._start_capture()

            logger.info(f"IDS Resolution updated to {self.width}x{self.height}")

    def _start_dark_ref_capture(self, *args: Any, **kwargs: Any) -> None:
        """
        Start an automated sequence to capture and average dark reference frames.
        Temporarily switches to Trigger mode and suppresses regular output emissions.
        """
        self.dark_ref_accumulator = None
        self.dark_ref_counter = 0
        self.collecting_dark_ref = True
        self._suppress_emission = True
        logger.info(f"Starting dark reference capture ({self.dark_ref_n_images} frames)")

        original_mode = dpg.get_value(self.trigger_tag) if dpg.does_item_exist(self.trigger_tag) else "Video"

        # Switch to Trigger mode to control capture manually
        if original_mode != "Trigger":
            self._set_trigger(None, "Trigger")
            if dpg.does_item_exist(self.trigger_tag):
                dpg.set_value(self.trigger_tag, "Trigger")

        def capture_sequence() -> None:
            time.sleep(0.1)
            expos_us = dpg.get_value(self.exposure_tag) if dpg.does_item_exist(self.exposure_tag) else 2000
            sleep_time = (expos_us / 1000000.0) + 0.1

            for _ in range(self.dark_ref_n_images):
                if not self.collecting_dark_ref:
                    break
                self.ids_cam.trigger()
                time.sleep(sleep_time)

            # Wait for last frame to be processed
            timeout = 2.0
            start = time.time()
            while self.collecting_dark_ref and time.time() - start < timeout:
                time.sleep(0.1)

            self.collecting_dark_ref = False

            # Restore original trigger mode
            if original_mode != "Trigger":
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
            - start (bool): Start live acquisition.
            - stop (bool): Stop live acquisition.
            - trigger (bool): Software trigger a single frame.
            - trigger_cam (bool/dict): Trigger camera, optionally with a temporary exposure time.
            - exposure (int): Set exposure time in microseconds.
            - gain (int): Set hardware gain (0-100).
            - pixel_clock (int): Set sensor pixel clock in MHz.
            - shutter (str): Set shutter mode ("Global", "Rolling", etc.).
            - trigger_mode (str): Set trigger mode ("Video" or "Trigger").
            - start_dark_ref (bool): Start dark reference acquisition.
            - n_frames (int): Set dark reference repetition count.
            - width (int) & height (int): Set AOI resolution.
            - ffc_enabled (bool): Enable or disable Flat-Field Correction.
            - capture_ffc (bool): Trigger capture of next frame as FFC reference.
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
                    old_exposure = dpg.get_value(self.exposure_tag) if dpg.does_item_exist(self.exposure_tag) else 2000
                    new_exposure = int(cmd["exposure"])
                    self._set_exposure(None, new_exposure)
                    if dpg.does_item_exist(self.exposure_tag):
                        dpg.set_value(self.exposure_tag, new_exposure)
                    time.sleep(0.1)
                    self.ids_cam.trigger()
                    time.sleep((new_exposure / 1000000.0) + 0.1)
                    self._set_exposure(None, old_exposure)
                    if dpg.does_item_exist(self.exposure_tag):
                        dpg.set_value(self.exposure_tag, old_exposure)

                threading.Thread(target=trigger_seq, daemon=True).start()
            else:
                self.ids_cam.trigger()

        if "trigger" in cmd:
            self.ids_cam.trigger()

        if "gain" in cmd:
            val = int(cmd["gain"])
            self._set_gain(None, val)
            if dpg.does_item_exist(self.gain_tag):
                dpg.set_value(self.gain_tag, val)

        if "exposure" in cmd and "trigger_cam" not in cmd and "start_dark_ref" not in cmd:
            val = int(cmd["exposure"])
            self._set_exposure(None, val)
            if dpg.does_item_exist(self.exposure_tag):
                dpg.set_value(self.exposure_tag, val)

        if "pixel_clock" in cmd:
            self._set_pixel_clock(None, int(cmd["pixel_clock"]))

        if "shutter" in cmd:
            self._set_shutter(None, str(cmd["shutter"]))

        if "start_dark_ref" in cmd:
            if "n_frames" in cmd:
                self.dark_ref_n_images = int(cmd["n_frames"])
            self._start_dark_ref_capture()

        if "trigger_mode" in cmd:
            mode = cmd["trigger_mode"]
            self._set_trigger(None, mode)
            if dpg.does_item_exist(self.trigger_tag):
                dpg.set_value(self.trigger_tag, mode)

        if "width" in cmd and "height" in cmd:
            self.width = int(cmd["width"])
            self.height = int(cmd["height"])
            if dpg.does_item_exist(self.width_tag):
                dpg.set_value(self.width_tag, self.width)
            if dpg.does_item_exist(self.height_tag):
                dpg.set_value(self.height_tag, self.height)
            self._apply_aoi()

        if "ffc_enabled" in cmd:
            self.ffc_enabled = bool(cmd["ffc_enabled"])
            if dpg.does_item_exist(self.ffc_enabled_tag):
                dpg.set_value(self.ffc_enabled_tag, self.ffc_enabled)

        if "capture_ffc" in cmd and cmd["capture_ffc"]:
            self._request_ffc_capture()

    def on_close(self) -> None:
        """
        Stop acquisition and release camera hardware resources upon window closure.
        """
        self._stop_capture()
        if hasattr(self, "ids_cam") and getattr(self.ids_cam, "defined", False):
            try:
                self.ids_cam.close()
            except Exception:
                pass
        super().on_close()

    def close(self) -> None:
        """
        Stop acquisition and release camera hardware resources upon window closing.
        """
        self.on_close()
        super().close()


EXPORTED_CLASS = IDS_win
EXPORTED_NAME = "IDS Camera"
