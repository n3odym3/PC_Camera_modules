from __future__ import annotations
from typing import Any, Dict, Tuple, Optional
import threading
import queue
import time
import os
import json
import numpy as np
import cv2
import dearpygui.dearpygui as dpg
from loguru import logger

from core.window_base import WindowBase
from core.input_output_types import IOTypes
from config.display_scaling import display_scaling
from core.config_manager import config
from core.app_state import app_state
from .Alvium import Alvium_cam


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

    description: str = "Control Allied Vision Alvium cameras. Supports exposure, gain, white balance, and trigger modes."

    init_btn_tag: str
    trigger_tag: str
    start_btn_tag: str
    exposure_tag: str
    gain_tag: str
    wb_r_tag: str
    wb_b_tag: str
    binning_tag: str
    format_tag: str
    roi_w_tag: str
    roi_h_tag: str
    sat_enabled_tag: str
    sat_factor_tag: str
    gamma_enabled_tag: str
    gamma_val_tag: str
    ffc_enabled_tag: str
    ffc_status_tag: str

    cam_running: bool
    cam_thread: Optional[threading.Thread]
    alvium_cam: Alvium_cam

    ffc_enabled: bool
    ffc_matrix_mono: Optional[np.ndarray]
    ffc_matrix_rgb: Optional[np.ndarray]
    ffc_gain_mono: Optional[np.ndarray]
    ffc_gain_rgb: Optional[np.ndarray]
    _capture_ffc_next: bool

    sat_enabled: bool
    sat_factor: float
    gamma_enabled: bool
    gamma_val: float
    _gamma_lut_8: Optional[np.ndarray]
    _gamma_lut_16: Optional[np.ndarray]

    def __init__(
        self,
        label: str = "Alvium Camera",
        win_width: int = display_scaling.scale(300),
        win_height: int = display_scaling.scale(400),
        pos: Tuple[int, int] = (10, 10),
        uuid: Optional[str] = None,
        outputs: Optional[Dict[str, Any]] = None,
        visible: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the Alvium Camera window module.

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
        )

        self.init_btn_tag = f"alvium_init_btn_{self.UUID}"
        self.trigger_tag = f"alvium_trigger_{self.UUID}"
        self.start_btn_tag = f"alvium_start_btn_{self.UUID}"
        self.exposure_tag = f"alvium_exposure_{self.UUID}"
        self.gain_tag = f"alvium_gain_{self.UUID}"
        self.wb_r_tag = f"alvium_wb_r_{self.UUID}"
        self.wb_b_tag = f"alvium_wb_b_{self.UUID}"
        self.binning_tag = f"alvium_binning_{self.UUID}"
        self.format_tag = f"alvium_format_{self.UUID}"
        self.roi_w_tag = f"alvium_roi_w_{self.UUID}"
        self.roi_h_tag = f"alvium_roi_h_{self.UUID}"
        self.sat_enabled_tag = f"alvium_sat_enabled_{self.UUID}"
        self.sat_factor_tag = f"alvium_sat_factor_{self.UUID}"
        self.gamma_enabled_tag = f"alvium_gamma_enabled_{self.UUID}"
        self.gamma_val_tag = f"alvium_gamma_val_{self.UUID}"
        self.ffc_enabled_tag = f"alvium_ffc_enabled_{self.UUID}"
        self.ffc_status_tag = f"alvium_ffc_status_{self.UUID}"

        self.cam_running = False
        self.cam_thread = None
        self.alvium_cam = Alvium_cam()

        self.ffc_enabled = False
        self.ffc_matrix_mono = None
        self.ffc_matrix_rgb = None
        self.ffc_gain_mono = None
        self.ffc_gain_rgb = None
        self._capture_ffc_next = False

        self._persistent_fields = ["label"]
        self.sat_enabled = False
        self.sat_factor = 1.0
        self.gamma_enabled = False
        self.gamma_val = 1.0
        self._gamma_lut_8 = None
        self._gamma_lut_16 = None
        self.accepted_input_types = [IOTypes.CMD_DICT]
        self.outputs = {
            "Frame": IOTypes.FRAME,
        }
        self.connections = {k: [] for k in self.outputs}

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
                    items=["Video", "Trigger"],
                    default_value="Video",
                    tag=self.trigger_tag,
                    callback=self._set_trigger,
                    width=display_scaling.scale(100),
                )
                dpg.add_button(
                    label="Trigger",
                    callback=lambda *args: self.alvium_cam.trigger(),
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
                    width=display_scaling.scale(100),
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
                width=display_scaling.scale(100),
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
                width=display_scaling.scale(150),
            )

            dpg.add_combo(
                label="Binning",
                items=["1", "2", "4", "8"],
                default_value="1",
                tag=self.binning_tag,
                callback=self._set_binning,
                width=display_scaling.scale(100),
            )

            dpg.add_separator()
            dpg.add_text("White Balance")
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Auto Once", callback=self._auto_white_balance
                )

            dpg.add_drag_float(
                label="WB Red",
                default_value=1.0,
                min_value=0.1,
                max_value=10.0,
                tag=self.wb_r_tag,
                callback=self._set_manual_wb,
                width=display_scaling.scale(70),
                speed=0.01,
            )
            dpg.add_drag_float(
                label="WB Blue",
                default_value=1.0,
                min_value=0.1,
                max_value=10.0,
                tag=self.wb_b_tag,
                callback=self._set_manual_wb,
                width=display_scaling.scale(70),
                speed=0.01,
            )

            dpg.add_separator()
            dpg.add_text("ROI / Resolution")
            with dpg.group(horizontal=True):
                dpg.add_input_int(
                    label="W",
                    tag=self.roi_w_tag,
                    step=-1,
                    width=display_scaling.scale(70),
                )
                dpg.add_input_int(
                    label="H",
                    tag=self.roi_h_tag,
                    step=-1,
                    width=display_scaling.scale(70),
                )
                dpg.add_button(label="Apply", callback=self._apply_roi)
                dpg.add_button(label="Max", callback=self._max_roi)

            dpg.add_separator()
            with dpg.collapsing_header(
                label="Image Processing", default_open=True
            ):
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
                    width=display_scaling.scale(150),
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
                    width=display_scaling.scale(150),
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
        Asynchronously initialize the Alvium camera connection, restore saved settings,
        and start frame acquisition.
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
            if self.alvium_cam.init():
                self.load_settings()
                if self.alvium_cam.had_error:
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
                    logger.error("Alvium camera configuration failed during settings application")
                    return

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

                logger.success("Alvium camera initialized")
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
                logger.error("Alvium camera initialization failed")

        threading.Thread(target=init_task, daemon=True).start()

    def _start_capture(
        self,
        sender: Optional[str] = None,
        app_data: Any = None,
        user_data: Any = None,
        *args: Any,
    ) -> None:
        """
        Start the background acquisition thread and initiate camera streaming.
        """
        if not self.alvium_cam.defined or self.alvium_cam.had_error:
            logger.error("Cannot start capture: Alvium camera is in an error state or is not initialized.")
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
            dpg.bind_item_theme(self.start_btn_tag, 0)

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
        Stop continuous frame acquisition and reset Start button styling.
        """
        if not self.cam_running:
            return
        self.cam_running = False
        if self.cam_thread:
            self.cam_thread.join(timeout=1.0)
        dpg.bind_item_theme(self.start_btn_tag, 0)

    def _apply_rotation(self, frame: np.ndarray) -> np.ndarray:
        """
        Rotate frame based on camera configuration (Cameras.Alvium.rotation).

        Args:
            frame: Input image array.

        Returns:
            np.ndarray: Rotated image array.
        """
        angle = config.get("Cameras", {}).get("Alvium", {}).get("rotation", 0)
        if angle == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def _trigger_outputs(self, frame: np.ndarray) -> None:
        """
        Process the acquired frame (rotation, FFC, saturation, gamma) and dispatch to outputs.

        Args:
            frame: Raw camera frame from frame queue.
        """
        frame = self._apply_rotation(frame)

        if self._capture_ffc_next:
            self._capture_ffc_next = False
            self._process_ffc_reference(frame)

        if self.ffc_enabled:
            try:
                is_color = len(frame.shape) == 3
                gain = self.ffc_gain_rgb if is_color else self.ffc_gain_mono

                if gain is not None:
                    dtype = frame.dtype
                    if frame.shape[:2] != gain.shape[:2]:
                        gain = cv2.resize(gain, (frame.shape[1], frame.shape[0]))

                    if is_color:
                        frame = frame.astype(np.float32) * np.expand_dims(gain, axis=2)
                    else:
                        frame = frame.astype(np.float32) * gain

                    is_12bit = "12" in self.alvium_cam.processing_mode
                    max_clip = 4095.0 if is_12bit else (65535.0 if dtype == np.uint16 else 255.0)
                    frame = np.clip(frame, 0, max_clip).astype(dtype)
            except Exception as e:
                logger.error(f"FFC application failed: {e}")

        if self.sat_enabled and len(frame.shape) == 3 and frame.shape[2] >= 3:
            try:
                is_16bit = frame.dtype == np.uint16
                is_12bit = "12" in self.alvium_cam.processing_mode
                divisor = 4095.0 if is_12bit else (65535.0 if is_16bit else 255.0)

                hsv = cv2.cvtColor(frame.astype(np.float32) / divisor, cv2.COLOR_BGR2HSV)
                hsv[:, :, 1] *= self.sat_factor
                hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 1.0)

                frame_float = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                frame = (frame_float * divisor).clip(0, divisor).astype(frame.dtype)
            except Exception as e:
                logger.error(f"Saturation adjustment failed: {e}")

        if self.gamma_enabled:
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
                logger.error(f"Gamma correction failed: {e}")

        data = (frame, "Alvium")
        for output_key, connected_modules in self.connections.items():
            for module in connected_modules:
                try:
                    module.input_cb(data)
                except Exception as e:
                    logger.error(f"Error sending frame to {module}: {e}")

    def _update_gamma(self, sender: Any, app_data: Any, user_data: Any = None, *args: Any, **kwargs: Any) -> None:
        """
        Recompute 8-bit and 16-bit lookup tables for fast gamma correction.

        Args:
            app_data: Gamma exponent value (0.1 to 3.0).
        """
        self.gamma_val = float(app_data)
        inv_gamma = 1.0 / max(0.1, self.gamma_val)

        indices_8 = np.arange(256, dtype=np.float32)
        self._gamma_lut_8 = np.clip(((indices_8 / 255.0) ** inv_gamma) * 255.0, 0, 255).astype(np.uint8)

        is_12bit = "12" in self.alvium_cam.processing_mode
        divisor = 4095.0 if is_12bit else 65535.0

        indices_16 = np.arange(65536, dtype=np.float32)
        self._gamma_lut_16 = np.clip(((indices_16 / divisor) ** inv_gamma) * divisor, 0, divisor).astype(np.uint16)

    def _request_ffc_capture(self, *args: Any) -> None:
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
                f_path = os.path.join(os.path.dirname(__file__), "alvium_ffc_rgb.npy")
            else:
                self.ffc_matrix_mono = matrix
                self.ffc_gain_mono = gain
                f_path = os.path.join(os.path.dirname(__file__), "alvium_ffc_mono.npy")

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

    def _set_trigger(self, sender: Any, app_data: Any, user_data: Any = None, *args: Any) -> None:
        """
        Set acquisition trigger mode ("Video" for continuous, "Trigger" for software/hardware trigger).
        """
        if self.alvium_cam.defined:
            self.alvium_cam.set_trigger_mode(app_data)

    def _set_exposure(self, sender: Any, app_data: Any, user_data: Any = None, *args: Any) -> None:
        """
        Set sensor exposure time.

        Args:
            app_data: Exposure time in milliseconds.
        """
        if self.alvium_cam.defined:
            exposure_us = float(app_data) * 1000.0
            self.alvium_cam.set_exposure(exposure_us)

    def _set_gain(self, sender: Any, app_data: Any, user_data: Any = None, *args: Any) -> None:
        """
        Set sensor hardware gain in dB (0-24).
        """
        if self.alvium_cam.defined:
            self.alvium_cam.set_gain(float(app_data))

    def _set_manual_wb(self, sender: Any, app_data: Any, user_data: Any = None, *args: Any) -> None:
        """
        Set manual white balance ratios for Red and Blue color channels.
        """
        if self.alvium_cam.defined:
            r = dpg.get_value(self.wb_r_tag)
            b = dpg.get_value(self.wb_b_tag)
            self.alvium_cam.set_manual_white_balance(r, b)

    def _set_format(self, sender: Any, app_data: Any, user_data: Any = None, *args: Any) -> None:
        """
        Set pixel format and update bit-depth dependent processing lookup tables.
        """
        if self.alvium_cam.defined:
            self.alvium_cam.set_format(app_data)
            self._update_gamma(None, self.gamma_val)

    def _set_binning(self, sender: Any, app_data: Any, user_data: Any = None, *args: Any) -> None:
        """
        Configure pixel binning factor and reset ROI to the new sensor maximum.
        """
        if self.alvium_cam.defined:
            self.alvium_cam.set_binning(app_data)
            self._max_roi()

    def _apply_roi(self, *args: Any) -> None:
        """
        Apply requested width and height to sensor Region of Interest (ROI).
        """
        if self.alvium_cam.defined:
            w = dpg.get_value(self.roi_w_tag)
            h = dpg.get_value(self.roi_h_tag)
            self.alvium_cam.set_roi(w, h)
            actual_w, actual_h = self.alvium_cam.get_roi()
            if actual_w > 0 and actual_h > 0:
                dpg.set_value(self.roi_w_tag, actual_w)
                dpg.set_value(self.roi_h_tag, actual_h)

    def _max_roi(self, *args: Any) -> None:
        """
        Reset sensor Region of Interest (ROI) to maximum available resolution.
        """
        if self.alvium_cam.defined:
            w, h = self.alvium_cam.get_max_roi()
            dpg.set_value(self.roi_w_tag, w)
            dpg.set_value(self.roi_h_tag, h)
            self.alvium_cam.set_roi(w, h)

    def _auto_exposure(self, *args: Any) -> None:
        """
        Execute one-shot auto exposure calibration and pull updated exposure to UI.
        """
        self.alvium_cam.auto_exposure()

        def pull_values() -> None:
            time.sleep(1.0)
            if self.alvium_cam.defined:
                expos_us = self.alvium_cam.get_exposure()
                dpg.set_value(self.exposure_tag, round(expos_us / 1000.0, 2))

        threading.Thread(target=pull_values, daemon=True).start()

    def _auto_white_balance(self, *args: Any) -> None:
        """
        Execute one-shot auto white balance calibration and pull updated WB to UI.
        """
        self.alvium_cam.auto_white_balance()

        def pull_values() -> None:
            time.sleep(1.0)
            if self.alvium_cam.defined:
                r, b = self.alvium_cam.get_white_balance()
                dpg.set_value(self.wb_r_tag, round(r, 2))
                dpg.set_value(self.wb_b_tag, round(b, 2))

        threading.Thread(target=pull_values, daemon=True).start()

    def save_settings(self, *args: Any) -> None:
        """
        Serialize all active camera and processing parameters to alvium_settings.json.
        """
        try:
            f = os.path.join(os.path.dirname(__file__), "alvium_settings.json")
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
            with open(f, "w") as fh:
                json.dump(data, fh, indent=4)
            logger.success("Alvium settings saved to alvium_settings.json")
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")

    def load_settings(self, *args: Any) -> None:
        """
        Load persisted parameters from alvium_settings.json and apply to hardware and UI.
        """
        try:
            f = os.path.join(os.path.dirname(__file__), "alvium_settings.json")
            if os.path.exists(f):
                with open(f, "r") as fh:
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
                self.alvium_cam.set_roi(dpg.get_value(self.roi_w_tag), dpg.get_value(self.roi_h_tag))

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

                p_mono = os.path.join(os.path.dirname(__file__), "alvium_ffc_mono.npy")
                p_rgb = os.path.join(os.path.dirname(__file__), "alvium_ffc_rgb.npy")

                status_parts = []
                if os.path.exists(p_mono):
                    try:
                        self.ffc_matrix_mono = np.load(p_mono)
                        self.ffc_gain_mono = 1.0 / self.ffc_matrix_mono
                        status_parts.append("Mono")
                    except Exception:
                        pass
                if os.path.exists(p_rgb):
                    try:
                        self.ffc_matrix_rgb = np.load(p_rgb)
                        self.ffc_gain_rgb = 1.0 / self.ffc_matrix_rgb
                        status_parts.append("RGB")
                    except Exception:
                        pass

                if status_parts and dpg.does_item_exist(self.ffc_status_tag):
                    dpg.set_value(self.ffc_status_tag, f"OK ({'+'.join(status_parts)})")
                    dpg.configure_item(self.ffc_status_tag, color=(100, 255, 100))

                logger.info("Restored Alvium settings from config")
        except Exception as e:
            logger.error(f"Failed to load settings: {e}")

    def is_ready(self) -> bool:
        """
        Check if the camera module is ready to accept commands or frames.

        Returns:
            bool: Always True for the camera window.
        """
        return True

    def input_cb(self, data: Any = None, *args: Any, **kwargs: Any) -> None:
        """
        Handle incoming command dictionary from connected modules or automation scripts.

        Supported command keys:
            - init (bool): Initialize camera hardware connection.
            - start (bool): Start continuous acquisition.
            - stop (bool): Stop acquisition.
            - trigger (bool): Trigger a single frame.
            - trigger_cam (bool/dict): Trigger a frame, optionally accepting an `exposure` override.
            - trigger_mode (str): Set trigger mode ("Video" or "Trigger").
            - exposure (float/int): Set exposure time in milliseconds.
            - gain (float/int): Set gain in dB (0-24).
            - binning (str): Set binning factor ("1", "2", "4", "8").
            - format (str): Set pixel format.
            - wb_r, wb_b (float): Set manual white balance ratios.
            - roi_w, roi_h (int): Set Region of Interest dimensions.
            - save_settings (bool): Save current configuration to file.
            - load_settings (bool): Reload configuration from file.
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
                if isinstance(cmd, dict) and "exposure" in cmd:
                    def trigger_seq() -> None:
                        old_exposure = dpg.get_value(self.exposure_tag)
                        new_exposure = float(cmd["exposure"])
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

            if "trigger" in cmd:
                self.alvium_cam.trigger()

            if "trigger_mode" in cmd:
                mode = cmd["trigger_mode"]
                self._set_trigger(None, mode)
                dpg.set_value(self.trigger_tag, mode)

            if "exposure" in cmd and "trigger_cam" not in cmd:
                val_ms = float(cmd["exposure"])
                self._set_exposure(None, val_ms)
                dpg.set_value(self.exposure_tag, val_ms)

            if "gain" in cmd:
                val = float(cmd["gain"])
                self._set_gain(None, val)
                dpg.set_value(self.gain_tag, val)

            if "binning" in cmd:
                val_bin = str(cmd["binning"])
                self._set_binning(None, val_bin)
                dpg.set_value(self.binning_tag, val_bin)

            if "format" in cmd:
                val_fmt = str(cmd["format"])
                self._set_format(None, val_fmt)
                dpg.set_value(self.format_tag, val_fmt)

            if "wb_r" in cmd or "wb_b" in cmd:
                if "wb_r" in cmd:
                    dpg.set_value(self.wb_r_tag, float(cmd["wb_r"]))
                if "wb_b" in cmd:
                    dpg.set_value(self.wb_b_tag, float(cmd["wb_b"]))
                self._set_manual_wb(None, None)

            if "roi_w" in cmd and "roi_h" in cmd:
                dpg.set_value(self.roi_w_tag, int(cmd["roi_w"]))
                dpg.set_value(self.roi_h_tag, int(cmd["roi_h"]))
                self._apply_roi()

            if "save_settings" in cmd and cmd["save_settings"]:
                self.save_settings()

            if "load_settings" in cmd and cmd["load_settings"]:
                self.load_settings()

    def close(self) -> None:
        """
        Stop acquisition and release camera hardware resources upon window closing.
        """
        self._stop_capture()
        self.alvium_cam.close()
        super().close()


EXPORTED_CLASS = Alvium_win
EXPORTED_NAME = "Alvium Camera"
