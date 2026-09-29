# Hamamatsu Camera Module

Controls **Hamamatsu** scientific CMOS (sCMOS) cameras using the `pyDCAM` driver wrapper and DCAM-API. This module provides continuous video streaming, hardware/software triggering, real-time dark reference correction, hardware binning, readout speed selection, and native 16-bit monochrome frame output.

---

## Technical Specifications

- **Accepted Inputs**: `In` (Type: `CMD_DICT` for remote automation and control)
- **Outputs**: `16bit frame` (Type: `FRAME16`, emits `(np.ndarray, "Hamamatsu")` containing `uint16` monochrome frames)
- **Default Resolution**: 2048x2048 native (sensor dependent, scalable via AOI subarray and binning)

---

## Key Features & Controls

- **Live Streaming & Trigger Modes**:
  - `Video`: Continuous internal frame acquisition.
  - `Trigger`: Hardware/external trigger (Rising edge polarity, edge active, global reset exposure).
  - `Trigger Software`: Software trigger dispatched on demand.
- **Exposure & Readout**:
  - Exposure time configured in microseconds (µs), converted to seconds for the DCAM API.
  - Readout speed selection (`Fastest` or `Slowest`) to balance frame rate and readout noise.
- **Hardware Binning**:
  - Select between `1x1`, `2x2`, and `4x4` hardware pixel binning with automatic sensor dimension recalculation.
- **Dark Reference Calibration**:
  - Accumulates and averages a configurable number of reference frames (`N Ref rep`).
  - Subtracts background noise in real-time.
  - Applies dynamic offset correction for pixels below the `Dark threshold`.
- **Area of Interest (AOI) / Subarray**:
  - Dynamically update sensor width and height (clamped to even pixel dimensions) with automatic circular buffer reallocation.
- **Rotation**:
  - Configurable image orientation via `config.Cameras.Hamamatsu.rotation` (0°, 90°, 180°, 270°).

---

## Remote Control API (`CMD_DICT`)

The module can be automated by sending dictionary commands to its input:

| Command Key | Type | Description |
| :--- | :--- | :--- |
| `init` | `bool` | Initializes the camera hardware connection and starts acquisition. |
| `start` | `bool` | Starts continuous frame acquisition. |
| `stop` | `bool` | Stops live acquisition. |
| `trigger` | `bool` | Dispatches a software trigger for a single frame. |
| `trigger_cam` | `bool` / `dict` | Triggers a frame, optionally accepting a temporary `exposure` override. |
| `exposure` | `int` | Sets the sensor exposure time in microseconds (µs). |
| `binning` | `int` | Sets hardware binning factor (`1`, `2`, or `4`). |
| `readout_speed` | `str` | Sets sensor readout speed (`"Fastest"` or `"Slowest"`). |
| `trigger_mode` | `str` | Selects acquisition mode: `"Video"`, `"Trigger"`, or `"Trigger Software"`. |
| `start_dark_ref`| `bool` | Initiates the dark reference capture sequence. |
| `n_frames` | `int` | Sets the number of frames to average for dark reference. |
| `width`, `height` | `int` | Adjusts sensor AOI dimensions (even pixel values required). |
