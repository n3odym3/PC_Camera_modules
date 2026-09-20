# Alvium Camera Module

Controls **Allied Vision Alvium** industrial cameras using the `vmbpy` SDK. This module provides continuous video streaming, software and hardware triggering, white balance calibration, multi-format pixel streaming, flat-field correction, and real-time color processing.

---

## Technical Specifications

- **Accepted Inputs**: `In` (Type: `CMD_DICT` for remote automation and control)
- **Outputs**: `Frame` (Type: `FRAME`, emits `(np.ndarray, "Alvium")` with `uint8` or `uint16` image arrays)
- **Supported Pixel Formats**:
  - `RGB8 (ISP)`: 8-bit color frame processed by the on-camera Image Signal Processor.
  - `Mono12`: 12-bit grayscale frame (in `uint16` container).
  - `BayerGR12 (Raw)`: 12-bit raw Bayer mosaic frame.
  - `BayerGR12 (RGB)`: 12-bit software debayered RGB frame.
  - `BayerGR12 (Red / Green / Blue)`: Individual color extraction from Bayer mosaic.

---

## Key Features & Controls

- **Acquisition & Trigger Modes**: Toggle between continuous live acquisition (`Video`) and software/hardware triggered capture (`Trigger`).
- **Exposure & Gain**:
  - Fine-tune exposure time in milliseconds (ms) with an **Auto Once** calibration trigger.
  - Set hardware analog gain from 0 to 24 dB.
- **White Balance Tuning**:
  - Adjust Red and Blue channel gain multipliers (0.1 to 10.0).
  - One-shot **Auto Once** automatic white balance balance button.
- **ROI & Binning**:
  - Pixel binning factors: 1x, 2x, 4x, 8x for high-speed or low-light sensitivity.
  - Custom Region of Interest (ROI) width and height with single-click **Max** reset.
- **Image Processing**:
  - **Saturation**: Real-time HSV color saturation enhancement (0.0 to 3.0x).
  - **Gamma**: Vectorized non-linear tone reproduction (gamma curve 0.1 to 3.0).
  - **Flat-Field Correction (FFC)**: Compute and apply grayscale normalization matrices to eliminate vignetting and uneven sensor illumination (persisted to `alvium_ffc_mono.npy` and `alvium_ffc_rgb.npy`).
- **Settings Persistence**: Save and reload full camera configuration to `alvium_settings.json`.

---

## Remote Control API (`CMD_DICT`)

The module can be automated by sending dictionary commands to its input:

| Command Key | Type | Description |
| :--- | :--- | :--- |
| `init` | `bool` | Initializes the camera hardware connection. |
| `start` | `bool` | Starts continuous frame acquisition. |
| `stop` | `bool` | Stops live acquisition. |
| `trigger` | `bool` | Dispatches a software trigger for a single frame. |
| `trigger_cam` | `bool` / `dict` | Triggers a frame, optionally accepting an `exposure` override in ms. |
| `trigger_mode` | `str` | Selects acquisition mode: `"Video"` or `"Trigger"`. |
| `exposure` | `float` / `int` | Sets the sensor exposure time in milliseconds (ms). |
| `gain` | `float` / `int` | Sets the sensor hardware gain in dB (0–24). |
| `binning` | `str` | Sets pixel binning factor (`"1"`, `"2"`, `"4"`, `"8"`). |
| `format` | `str` | Sets pixel format (e.g. `"RGB8 (ISP)"`, `"Mono12"`). |
| `wb_r`, `wb_b` | `float` | Sets manual Red and Blue white balance ratios. |
| `roi_w`, `roi_h` | `int` | Sets sensor Region of Interest (ROI) dimensions. |
| `save_settings` | `bool` | Saves active parameters to `alvium_settings.json`. |
| `load_settings` | `bool` | Reloads parameters from `alvium_settings.json`. |
