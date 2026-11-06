from __future__ import annotations

import json
import os
import queue
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import dearpygui.dearpygui as dpg
import numpy as np
from loguru import logger

from config.display_scaling import display_scaling
from core.app_state import app_state
from core.config_manager import config
from core.input_output_types import IOTypes
from core.window_base import WindowBase
from .Alvium import Alvium_cam

_MODULE_DIR = os.path.dirname(__file__)
_SETTINGS_PATH = os.path.join(_MODULE_DIR, "alvium_settings.json")
_FFC_MONO_PATH = os.path.join(_MODULE_DIR, "alvium_ffc_mono.npy")
_FFC_RGB_PATH = os.path.join(_MODULE_DIR, "alvium_ffc_rgb.npy")
_ROTATION_MAP = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


class Alvium_win(WindowBase):
    """
    Window module for controlling Allied Vision Alvium industrial cameras via vmbpy.

    Features:
        - Continuous live acquisition ("Video") and software/hardware ("Trigger") modes.
        - Multi-format acquisition: RGB8 (ISP), Mono12, BayerGR12 (Raw/RGB/R/G/B).
        - Dynamic Exposure time (ms) with optional one-shot Auto Exposure.
        - Hardware gain adjustment (0-24 dB).
        - Manual white balance (Red/Blue channels) and one-shot Auto White Balance.
        - Pixel binning (1x, 2x, 4x, 8x) and Region of Interest (ROI) adjustments.
        - Flat-Field Correction (FFC) with separate Mono and RGB calibration matrices.
        - Real-time saturation control and vectorized gamma correction.
        - Remote automation control via dictionary commands (`IOTypes.CMD_DICT`).
    """

    label: str = "Alvium Camera"
    win_width: int = display_scaling.scale(320)
    win_height: int = display_scaling.scale(490)
    description: str = (
        "Control Allied Vision Alvium industrial cameras via vmbpy. "
        "Supports exposure, gain, white balance, binning, FFC, and trigger modes."
    )

    accepted_input_types: list[str] = [IOTypes.CMD_DICT]
    outputs: dict[str, str] = {
        "Frame": IOTypes.FRAME,
    }
    persistent_fields: list[str] = ["label"]

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the Alvium Camera window module.

        Args:
            **kwargs: Additional keyword arguments forwarded to WindowBase.
        """
        super().__init__(**kwargs)

        uid = self.UUID
        self.init_btn_tag: str = f"alvium_init_btn_{uid}"
        self.trigger_tag: str = f"alvium_trigger_{uid}"
        self.start_btn_tag: str = f"alvium_start_btn_{uid}"
        self.exposure_tag: str = f"alvium_exposure_{uid}"
        self.gain_tag: str = f"alvium_gain_{uid}"
        self.wb_r_tag: str = f"alvium_wb_r_{uid}"
        self.wb_b_tag: str = f"alvium_wb_b_{uid}"
        self.binning_tag: str = f"alvium_binning_{uid}"
        self.format_tag: str = f"alvium_format_{uid}"
        self.roi_w_tag: str = f"alvium_roi_w_{uid}"
        self.roi_h_tag: str = f"alvium_roi_h_{uid}"
        self.sat_enabled_tag: str = f"alvium_sat_enabled_{uid}"
        self.sat_factor_tag: str = f"alvium_sat_factor_{uid}"
        self.gamma_enabled_tag: str = f"alvium_gamma_enabled_{uid}"
        self.gamma_val_tag: str = f"alvium_gamma_val_{uid}"
        self.ffc_enabled_tag: str = f"alvium_ffc_enabled_{uid}"
        self.ffc_status_tag: str = f"alvium_ffc_status_{uid}"

        self.cam_running: bool = False
        self.cam_thread: Optional[threading.Thread] = None
        self.alvium_cam: Alvium_cam = Alvium_cam()

        self.ffc_enabled: bool = False
        self.ffc_matrix_mono: Optional[np.ndarray] = None
        self.ffc_matrix_rgb: Optional[np.ndarray] = None
        self.ffc_gain_mono: Optional[np.ndarray] = None
        self.ffc_gain_rgb: Optional[np.ndarray] = None
        self._ffc_gain_mono_cached: Optional[np.ndarray] = None
        self._ffc_gain_rgb_cached: Optional[np.ndarray] = None
        self._capture_ffc_next: bool = False

        self.sat_enabled: bool = False
        self.sat_factor: float = 1.0
        self.gamma_enabled: bool = False
        self.gamma_val: float = 1.0
        self._gamma_lut_8: Optional[np.ndarray] = None
        self._gamma_lut_16: Optional[np.ndarray] = None

        angle = config.get("Cameras", {}).get("Alvium", {}).get("rotation", 0)
        self._rotation_code: Optional[int] = _ROTATION_MAP.get(angle)

        self._theme_init: Optional[int] = None
        self._theme_success: Optional[int] = None
        self._theme_fail: Optional[int] = None

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
        """Construct DearPyGui widgets for Alvium camera control."""
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
                    width=scale(100),
                )
                dpg.add_button(
                    label="Trigger",
                    callback=lambda: self.alvium_cam.trigger(),
                )

            with dpg.group(horizontal=True):
                dpg.add_input_float(
                    label="Expos. (ms)",
                    default_value=1.0,
                    min_value=0.01,
                    max_value=1000,
                    format="%.2f",
                    step=-1,
                    tag=self.exposure_tag,
                    callback=self._set_exposure,
                    on_enter=True,
                    width=scale(100),
                )
                dpg.add_button(label="Auto Once", callback=self._auto_exposure)

            dpg.add_input_int(
                label="Gain (dB)",
                default_value=1,
                min_value=0,
                max_value=24,
                step=-1,
                tag=self.gain_tag,
                callback=self._set_gain,
                on_enter=True,
                width=scale(100),
            )

            dpg.add_combo(
                label="Format",
                items=[
                    "RGB8 (ISP)",
                    "Mono12",
                    "BayerGR12 (Raw)",
                    "BayerGR12 (RGB)",
                    "BayerGR12 (Red)",
                    "BayerGR12 (Green)",
                    "BayerGR12 (Blue)",
                ],
                default_value="RGB8 (ISP)",
                tag=self.format_tag,
                callback=self._set_format,
                width=scale(150),
            )

            dpg.add_combo(
                label="Binning",
                items=["1", "2", "4", "8"],
                default_value="1",
                tag=self.binning_tag,
                callback=self._set_binning,
                width=scale(100),
            )

            dpg.add_separator()
            dpg.add_text("White Balance")
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Auto Once",
                    callback=self._auto_white_balance,
                )

            dpg.add_drag_float(
                label="WB Red",
                default_value=1.0,
                min_value=0.1,
                max_value=10.0,
                tag=self.wb_r_tag,
                callback=self._set_manual_wb,
                width=scale(70),
                speed=0.01,
            )
            dpg.add_drag_float(
                label="WB Blue",
                default_value=1.0,
                min_value=0.1,
                max_value=10.0,
                tag=self.wb_b_tag,
                callback=self._set_manual_wb,
                width=scale(70),
                speed=0.01,
            )

            dpg.add_separator()
            dpg.add_text("ROI / Resolution")
            with dpg.group(horizontal=True):
                dpg.add_input_int(
                    label="W",
                    tag=self.roi_w_tag,
                    step=-1,
                    width=scale(70),
                )
                dpg.add_input_int(
                    label="H",
                    tag=self.roi_h_tag,
                    step=-1,
                    width=scale(70),
                )
                dpg.add_button(label="Apply", callback=self._apply_roi)
                dpg.add_button(label="Max", callback=self._max_roi)

            dpg.add_separator()
            with dpg.collapsing_header(label="Image Processing", default_open=True):
                dpg.add_checkbox(
                    label="Enable Saturation",
                    tag=self.sat_enabled_tag,
                    callback=lambda s, v: setattr(self, "sat_enabled", v),
                )
                dpg.add_slider_float(
                    label="Saturation Factor",
                    tag=self.sat_factor_tag,
                    default_value=1.0,
                    min_value=0.0,
                    max_value=3.0,
                    callback=lambda s, v: setattr(self, "sat_factor", v),
                    width=scale(150),
                )

                dpg.add_separator()

                dpg.add_checkbox(
                    label="Enable Gamma",
                    tag=self.gamma_enabled_tag,
                    callback=lambda s, v: setattr(self, "gamma_enabled", v),
                )
                dpg.add_slider_float(
                    label="Gamma Value",
                    tag=self.gamma_val_tag,
                    default_value=1.0,
                    min_value=0.1,
                    max_value=3.0,
                    callback=self._update_gamma,
                    width=scale(150),
                )

                dpg.add_separator()
                dpg.add_text("Flat Field Correction (FFC)")
                dpg.add_checkbox(
                    label="Enable FFC",
                    tag=self.ffc_enabled_tag,
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
            dpg.add_button(
                label="Save Settings (Config)",
                callback=self.save_settings,
                width=-1,
            )

    def update_permission(self) -> None:
        """Adjust module permissions and UI visibility based on the application mode."""
        is_user = getattr(app_state, "mode", "dev") == "user"
        if dpg.does_item_exist(self.winID):
            dpg.configure_item(self.winID, no_close=is_user)

    def _init_camera(
        self,
        sender: Any = None,
        app_data: Any = None,
        user_data: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Asynchronously initialize the Alvium camera connection, restore saved settings,
        and start frame acquisition.
        """
        self._ensure_themes()
        if dpg.does_item_exist(self.init_btn_tag):
            dpg.bind_item_theme(self.init_btn_tag, self._theme_init)

        def init_task() -> None:
            if self.alvium_cam.init():
                self.load_settings()
                if self.alvium_cam.had_error:
                    if dpg.does_item_exist(self.init_btn_tag):
                        dpg.bind_item_theme(self.init_btn_tag, self._theme_fail)
                    logger.error(f"[{self.label}] Configuration failed during settings application")
                    return

                if dpg.does_item_exist(self.init_btn_tag):
                    dpg.bind_item_theme(self.init_btn_tag, self._theme_success)

                logger.success(f"[{self.label}] Alvium camera initialized")
                self._start_capture()
            else:
                if dpg.does_item_exist(self.init_btn_tag):
                    dpg.bind_item_theme(self.init_btn_tag, self._theme_fail)
                logger.error(f"[{self.label}] Alvium camera initialization failed")

        threading.Thread(target=init_task, daemon=True).start()

    def _start_capture(
        self,
        sender: Any = None,
        app_data: Any = None,
        user_data: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Start the background acquisition thread and initiate camera streaming.
        """
        if not self.alvium_cam.defined or self.alvium_cam.had_error:
            logger.error(f"[{self.label}] Cannot start capture: camera is in error state or uninitialized.")
            return

        if self.cam_thread and self.cam_thread.is_alive():
            return

        self.cam_running = True

        self._ensure_themes()
        if dpg.does_item_exist(self.start_btn_tag):
            dpg.bind_item_theme(self.start_btn_tag, self._theme_success)

        self.alvium_cam.start_streaming()

        def capture_task() -> None:
            while self.cam_running and not self.alvium_cam.had_error:
                try:
                    frame = self.alvium_cam.frame_queue.get(timeout=0.01)
                    self._trigger_outputs(frame)
                except queue.Empty:
                    pass
            self.alvium_cam.stop_streaming()
            self.cam_running = False
            if dpg.does_item_exist(self.start_btn_tag):
                dpg.bind_item_theme(self.start_btn_tag, 0)

        self.cam_thread = threading.Thread(target=capture_task, daemon=True)
        self.cam_thread.start()

    def _stop_capture(
        self,
        sender: Any = None,
        app_data: Any = None,
        user_data: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Stop continuous frame acquisition and reset Start button styling.
        """
        if not self.cam_running:
            return
        self.cam_running = False
        if self.cam_thread:
            self.cam_thread.join(timeout=1.0)
        if dpg.does_item_exist(self.start_btn_tag):
            dpg.bind_item_theme(self.start_btn_tag, 0)

    def _apply_rotation(self, frame: np.ndarray) -> np.ndarray:
        """
        Rotate frame based on camera configuration (Cameras.Alvium.rotation).
        """
        if self._rotation_code is not None:
            return cv2.rotate(frame, self._rotation_code)
        return frame

    def _trigger_outputs(self, frame: np.ndarray) -> None:
        """
        Process the acquired frame (rotation, FFC, saturation, gamma) and dispatch to outputs.
        """
        frame = self._apply_rotation(frame)

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

                    is_12bit = "12" in self.alvium_cam.processing_mode
                    max_clip = 4095.0 if is_12bit else (65535.0 if dtype == np.uint16 else 255.0)
                    np.clip(frame_f, 0.0, max_clip, out=frame_f)
                    frame = frame_f.astype(dtype)
            except Exception as e:
                logger.error(f"[{self.label}] FFC application failed: {e}")

        if self.sat_enabled and self.sat_factor != 1.0 and frame.ndim == 3 and frame.shape[2] >= 3:
            try:
                is_16bit = frame.dtype == np.uint16
                is_12bit = "12" in self.alvium_cam.processing_mode
                divisor = 4095.0 if is_12bit else (65535.0 if is_16bit else 255.0)

                frame_f = frame.astype(np.float32)
                frame_f *= 1.0 / divisor
                hsv = cv2.cvtColor(frame_f, cv2.COLOR_BGR2HSV)
                sat_ch = hsv[:, :, 1]
                sat_ch *= self.sat_factor
                np.clip(sat_ch, 0.0, 1.0, out=sat_ch)

                frame_float = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                frame_float *= divisor
                np.clip(frame_float, 0.0, divisor, out=frame_float)
                frame = frame_float.astype(frame.dtype)
            except Exception as e:
                logger.error(f"[{self.label}] Saturation adjustment failed: {e}")

        if self.gamma_enabled and self.gamma_val != 1.0:
            try:
                if frame.dtype == np.uint8:
                    if self._gamma_lut_8 is None:
                        self._update_gamma(None, self.gamma_val)
                    if self._gamma_lut_8 is not None:
                        frame = cv2.LUT(frame, self._gamma_lut_8)
                else:
                    if self._gamma_lut_16 is None:
                        self._update_gamma(None, self.gamma_val)
                    if self._gamma_lut_16 is not None:
                        frame = self._gamma_lut_16[frame]
            except Exception as e:
                logger.error(f"[{self.label}] Gamma correction failed: {e}")

        data = (frame, "Alvium")
        for output_key, connected_modules in self.connections.items():
            for module in connected_modules:
                try:
                    module.input_cb(data, data_type=IOTypes.FRAME)
                except Exception as e:
                    logger.error(f"[{self.label}] Error sending frame to {getattr(module, 'label', 'module')}: {e}")

    def _update_gamma(
        self,
        sender: Any = None,
        app_data: Any = None,
        user_data: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Recompute 8-bit and 16-bit lookup tables for fast gamma correction.
        """
        if app_data is not None:
            self.gamma_val = float(app_data)
        inv_gamma = 1.0 / max(0.1, self.gamma_val)

        indices_8 = np.arange(256, dtype=np.float32)
        self._gamma_lut_8 = np.clip(((indices_8 / 255.0) ** inv_gamma) * 255.0, 0, 255).astype(np.uint8)

        is_12bit = "12" in self.alvium_cam.processing_mode
        divisor = 4095.0 if is_12bit else 65535.0

        indices_16 = np.arange(65536, dtype=np.float32)
        self._gamma_lut_16 = np.clip(((indices_16 / divisor) ** inv_gamma) * divisor, 0, divisor).astype(np.uint16)

    def _request_ffc_capture(
        self,
        sender: Any = None,
        app_data: Any = None,
        user_data: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Flag the next incoming frame for Flat-Field Correction (FFC) reference calculation.
        """
        if not self.cam_running:
            logger.warning(f"[{self.label}] Camera must be running to capture FFC reference")
            return
        self._capture_ffc_next = True
        logger.info(f"[{self.label}] FFC capture requested - point camera at uniform white source")

    def _process_ffc_reference(self, frame: np.ndarray) -> None:
        """
        Calculate and persist flat-field normalization matrix from a uniform illumination frame.
        """
        try:
            is_color = len(frame.shape) == 3

            if is_color:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame.copy()

            gray_f = gray.astype(np.float32)
            gray_f = cv2.GaussianBlur(gray_f, (31, 31), 0)

            peak = float(np.max(gray_f))
            if peak == 0.0:
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
            logger.success(f"[{self.label}] FFC {'RGB' if is_color else 'Mono'} reference captured successfully")

        except Exception as e:
            logger.error(f"[{self.label}] FFC processing failed: {e}")
            if dpg.does_item_exist(self.ffc_status_tag):
                dpg.set_value(self.ffc_status_tag, "Error")
                dpg.configure_item(self.ffc_status_tag, color=(255, 100, 100))

    def _set_trigger(
        self,
        sender: Any = None,
        app_data: Any = None,
        user_data: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Set acquisition trigger mode ("Video" or "Trigger")."""
        if self.alvium_cam.defined:
            self.alvium_cam.set_trigger_mode(str(app_data or "Video"))

    def _set_exposure(
        self,
        sender: Any = None,
        app_data: Any = None,
        user_data: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Set sensor exposure time in milliseconds."""
        if self.alvium_cam.defined and app_data is not None:
            exposure_us = float(app_data) * 1000.0
            self.alvium_cam.set_exposure(exposure_us)

    def _set_gain(
        self,
        sender: Any = None,
        app_data: Any = None,
        user_data: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Set sensor hardware gain in dB (0-24)."""
        if self.alvium_cam.defined and app_data is not None:
            self.alvium_cam.set_gain(float(app_data))

    def _set_manual_wb(
        self,
        sender: Any = None,
        app_data: Any = None,
        user_data: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Set manual white balance ratios for Red and Blue color channels."""
        if self.alvium_cam.defined:
            r = dpg.get_value(self.wb_r_tag)
            b = dpg.get_value(self.wb_b_tag)
            self.alvium_cam.set_manual_white_balance(float(r), float(b))

    def _set_format(
        self,
        sender: Any = None,
        app_data: Any = None,
        user_data: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Set pixel format and update lookup tables."""
        if self.alvium_cam.defined and app_data:
            self.alvium_cam.set_format(str(app_data))
            self._update_gamma(None, self.gamma_val)

    def _set_binning(
        self,
        sender: Any = None,
        app_data: Any = None,
        user_data: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Configure pixel binning factor and reset ROI to new max."""
        if self.alvium_cam.defined and app_data:
            self.alvium_cam.set_binning(str(app_data))
            actual_w, actual_h = self.alvium_cam.get_roi()
            if actual_w > 0 and actual_h > 0:
                dpg.set_value(self.roi_w_tag, actual_w)
                dpg.set_value(self.roi_h_tag, actual_h)

    def _apply_roi(
        self,
        sender: Any = None,
        app_data: Any = None,
        user_data: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Apply requested width and height to sensor Region of Interest (ROI)."""
        if self.alvium_cam.defined:
            w = dpg.get_value(self.roi_w_tag)
            h = dpg.get_value(self.roi_h_tag)
            self.alvium_cam.set_roi(int(w), int(h))
            actual_w, actual_h = self.alvium_cam.get_roi()
            if actual_w > 0 and actual_h > 0:
                dpg.set_value(self.roi_w_tag, actual_w)
                dpg.set_value(self.roi_h_tag, actual_h)

    def _max_roi(
        self,
        sender: Any = None,
        app_data: Any = None,
        user_data: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Reset sensor Region of Interest (ROI) to maximum available resolution."""
        if self.alvium_cam.defined:
            w, h = self.alvium_cam.get_max_roi()
            dpg.set_value(self.roi_w_tag, w)
            dpg.set_value(self.roi_h_tag, h)
            self.alvium_cam.set_roi(w, h)

    def _auto_exposure(
        self,
        sender: Any = None,
        app_data: Any = None,
        user_data: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Execute one-shot auto exposure calibration and pull updated value."""
        self.alvium_cam.auto_exposure()

        def pull_values() -> None:
            time.sleep(1.0)
            if self.alvium_cam.defined:
                expos_us = self.alvium_cam.get_exposure()
                if dpg.does_item_exist(self.exposure_tag):
                    dpg.set_value(self.exposure_tag, round(expos_us / 1000.0, 2))

        threading.Thread(target=pull_values, daemon=True).start()

    def _auto_white_balance(
        self,
        sender: Any = None,
        app_data: Any = None,
        user_data: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Execute one-shot auto white balance calibration and pull updated values."""
        self.alvium_cam.auto_white_balance()

        def pull_values() -> None:
            time.sleep(1.0)
            if self.alvium_cam.defined:
                r, b = self.alvium_cam.get_white_balance()
                if dpg.does_item_exist(self.wb_r_tag):
                    dpg.set_value(self.wb_r_tag, round(r, 2))
                if dpg.does_item_exist(self.wb_b_tag):
                    dpg.set_value(self.wb_b_tag, round(b, 2))

        threading.Thread(target=pull_values, daemon=True).start()

    def save_settings(
        self,
        sender: Any = None,
        app_data: Any = None,
        user_data: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Serialize all active camera and processing parameters to alvium_settings.json."""
        try:
            data = {
                "exposure": dpg.get_value(self.exposure_tag),
                "gain": dpg.get_value(self.gain_tag),
                "format": dpg.get_value(self.format_tag),
                "binning": dpg.get_value(self.binning_tag),
                "roi_w": dpg.get_value(self.roi_w_tag),
                "roi_h": dpg.get_value(self.roi_h_tag),
                "wb_r": dpg.get_value(self.wb_r_tag),
                "wb_b": dpg.get_value(self.wb_b_tag),
                "sat_enabled": dpg.get_value(self.sat_enabled_tag),
                "sat_factor": dpg.get_value(self.sat_factor_tag),
                "gamma_enabled": dpg.get_value(self.gamma_enabled_tag),
                "gamma_val": dpg.get_value(self.gamma_val_tag),
                "ffc_enabled": dpg.get_value(self.ffc_enabled_tag),
            }
            with open(_SETTINGS_PATH, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=4)
            logger.success(f"[{self.label}] Settings saved to alvium_settings.json")
        except Exception as e:
            logger.error(f"[{self.label}] Failed to save settings: {e}")

    def load_settings(
        self,
        sender: Any = None,
        app_data: Any = None,
        user_data: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Load persisted parameters from alvium_settings.json and apply to hardware and UI."""
        try:
            angle = config.get("Cameras", {}).get("Alvium", {}).get("rotation", 0)
            self._rotation_code = _ROTATION_MAP.get(angle)

            if os.path.exists(_SETTINGS_PATH):
                with open(_SETTINGS_PATH, "r", encoding="utf-8") as fh:
                    data = json.load(fh)

                exposure_val = float(data.get("exposure", 1.0))
                dpg.set_value(self.exposure_tag, exposure_val)
                self._set_exposure(None, exposure_val)

                gain_val = float(data.get("gain", 1.0))
                dpg.set_value(self.gain_tag, gain_val)
                self._set_gain(None, gain_val)

                fmt = str(data.get("format", "RGB8 (ISP)"))
                dpg.set_value(self.format_tag, fmt)
                self.alvium_cam.set_format(fmt)

                binning = str(data.get("binning", "1"))
                dpg.set_value(self.binning_tag, binning)
                self.alvium_cam.set_binning(binning)

                rw, rh = self.alvium_cam.get_roi()
                dpg.set_value(self.roi_w_tag, data.get("roi_w", rw))
                dpg.set_value(self.roi_h_tag, data.get("roi_h", rh))
                self.alvium_cam.set_roi(
                    int(dpg.get_value(self.roi_w_tag)),
                    int(dpg.get_value(self.roi_h_tag)),
                )

                dpg.set_value(self.wb_r_tag, float(data.get("wb_r", 1.0)))
                dpg.set_value(self.wb_b_tag, float(data.get("wb_b", 1.0)))
                self._set_manual_wb(None, None)

                self.sat_enabled = bool(data.get("sat_enabled", False))
                dpg.set_value(self.sat_enabled_tag, self.sat_enabled)
                self.sat_factor = float(data.get("sat_factor", 1.0))
                dpg.set_value(self.sat_factor_tag, self.sat_factor)

                self.gamma_enabled = bool(data.get("gamma_enabled", False))
                dpg.set_value(self.gamma_enabled_tag, self.gamma_enabled)
                self.gamma_val = float(data.get("gamma_val", 1.0))
                dpg.set_value(self.gamma_val_tag, self.gamma_val)
                self._update_gamma(None, self.gamma_val)

                self.ffc_enabled = bool(data.get("ffc_enabled", False))
                dpg.set_value(self.ffc_enabled_tag, self.ffc_enabled)

                status_parts: list[str] = []
                if os.path.exists(_FFC_MONO_PATH):
                    try:
                        self.ffc_matrix_mono = np.load(_FFC_MONO_PATH)
                        self.ffc_gain_mono = 1.0 / self.ffc_matrix_mono
                        self._ffc_gain_mono_cached = self.ffc_gain_mono
                        status_parts.append("Mono")
                    except Exception:
                        pass
                if os.path.exists(_FFC_RGB_PATH):
                    try:
                        self.ffc_matrix_rgb = np.load(_FFC_RGB_PATH)
                        self.ffc_gain_rgb = 1.0 / self.ffc_matrix_rgb
                        self._ffc_gain_rgb_cached = np.expand_dims(self.ffc_gain_rgb, axis=2)
                        status_parts.append("RGB")
                    except Exception:
                        pass

                if status_parts and dpg.does_item_exist(self.ffc_status_tag):
                    dpg.set_value(self.ffc_status_tag, f"OK ({'+'.join(status_parts)})")
                    dpg.configure_item(self.ffc_status_tag, color=(100, 255, 100))

                logger.info(f"[{self.label}] Restored Alvium settings from config")
        except Exception as e:
            logger.error(f"[{self.label}] Failed to load settings: {e}")

    def is_ready(self) -> bool:
        """
        Check if the camera module is ready to accept commands or frames.
        """
        return True

    def input_cb(self, data: Any = None, *args: Any, **kwargs: Any) -> None:
        """
        Handle incoming command dictionary from connected modules or automation scripts.
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
            if cmd.get("init"):
                self._init_camera()

            if cmd.get("start"):
                self._start_capture()

            if cmd.get("stop"):
                self._stop_capture()

            if "trigger_cam" in cmd:
                trig_arg = cmd["trigger_cam"]
                if isinstance(trig_arg, dict) and "exposure" in trig_arg:
                    def trigger_seq() -> None:
                        old_exposure = float(dpg.get_value(self.exposure_tag))
                        new_exposure = float(trig_arg["exposure"])
                        self._set_exposure(None, new_exposure)
                        dpg.set_value(self.exposure_tag, new_exposure)
                        time.sleep(0.05)
                        self.alvium_cam.trigger()
                        time.sleep((new_exposure / 1000.0) + 0.05)
                        self._set_exposure(None, old_exposure)
                        dpg.set_value(self.exposure_tag, old_exposure)

                    threading.Thread(target=trigger_seq, daemon=True).start()
                else:
                    self.alvium_cam.trigger()

            if cmd.get("trigger"):
                self.alvium_cam.trigger()

            if "trigger_mode" in cmd:
                mode = str(cmd["trigger_mode"])
                self._set_trigger(None, mode)
                if dpg.does_item_exist(self.trigger_tag):
                    dpg.set_value(self.trigger_tag, mode)

            if "exposure" in cmd and "trigger_cam" not in cmd:
                val_ms = float(cmd["exposure"])
                self._set_exposure(None, val_ms)
                if dpg.does_item_exist(self.exposure_tag):
                    dpg.set_value(self.exposure_tag, val_ms)

            if "gain" in cmd:
                val_g = float(cmd["gain"])
                self._set_gain(None, val_g)
                if dpg.does_item_exist(self.gain_tag):
                    dpg.set_value(self.gain_tag, val_g)

            if "binning" in cmd:
                val_bin = str(cmd["binning"])
                self._set_binning(None, val_bin)
                if dpg.does_item_exist(self.binning_tag):
                    dpg.set_value(self.binning_tag, val_bin)

            if "format" in cmd:
                val_fmt = str(cmd["format"])
                self._set_format(None, val_fmt)
                if dpg.does_item_exist(self.format_tag):
                    dpg.set_value(self.format_tag, val_fmt)

            if "wb_r" in cmd or "wb_b" in cmd:
                if "wb_r" in cmd and dpg.does_item_exist(self.wb_r_tag):
                    dpg.set_value(self.wb_r_tag, float(cmd["wb_r"]))
                if "wb_b" in cmd and dpg.does_item_exist(self.wb_b_tag):
                    dpg.set_value(self.wb_b_tag, float(cmd["wb_b"]))
                self._set_manual_wb(None, None)

            if "roi_w" in cmd and "roi_h" in cmd:
                if dpg.does_item_exist(self.roi_w_tag):
                    dpg.set_value(self.roi_w_tag, int(cmd["roi_w"]))
                if dpg.does_item_exist(self.roi_h_tag):
                    dpg.set_value(self.roi_h_tag, int(cmd["roi_h"]))
                self._apply_roi()

            if cmd.get("save_settings"):
                self.save_settings()

            if cmd.get("load_settings"):
                self.load_settings()

    def on_close(self) -> None:
        """
        Stop acquisition and release camera hardware resources upon window closure.
        """
        self._stop_capture()
        if hasattr(self, "alvium_cam") and self.alvium_cam is not None:
            self.alvium_cam.close()
        super().on_close()

    def close(self) -> None:
        """
        Stop acquisition and release camera hardware resources upon window closing.
        """
        self._stop_capture()
        if hasattr(self, "alvium_cam") and self.alvium_cam is not None:
            self.alvium_cam.close()
        super().close()


EXPORTED_CLASS = Alvium_win
EXPORTED_NAME = "Alvium Camera"
