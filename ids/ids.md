# IDS Camera Module

Controls **IDS Imaging Development Systems** industrial cameras using the `pyueye` SDK. This module provides continuous video streaming, hardware/software triggering, real-time dark reference correction, and 12-bit monochrome frame output.

---

## Technical Specifications

- **Accepted Inputs**: `In` (Type: `CMD_DICT` for remote automation and control)
- **Outputs**: `12bit frame` (Type: `FRAME12`, emits `(np.ndarray, "IDS")` containing 16-bit container `uint16` monochrome frames)
- **Default Resolution**: Sensor native / AOI (e.g. 1280x1024)

---

## Key Features & Controls

- **Live Streaming & Trigger Modes**: Toggle between continuous live acquisition (`Video`) and external/software triggered capture (`Trigger`).
- **Exposure & Gain**: Fine-tune exposure time in microseconds (µs) and hardware gain (0–100).
- **Dark Reference Calibration**:
  - Accumulates and averages a configurable number of reference frames (`N Ref rep`).
  - Subtracts background noise in real-time.
  - Applies dynamic offset correction for pixels below the `Dark threshold`.
- **Area of Interest (AOI) / Resolution**: Dynamically update sensor width and height (clamped to even pixel dimensions) with automatic DMA memory reallocation.
- **Rotation**: Configurable image orientation via `config.Cameras.IDS.rotation` (0°, 90°, 180°, 270°).

---

## Remote Control API (`CMD_DICT`)

The module can be automated by sending dictionary commands to its input:

| Command Key | Type | Description |
| :--- | :--- | :--- |
| `init` | `bool` | Initializes the camera hardware connection. |
| `start` | `bool` | Starts continuous frame acquisition. |
| `stop` | `bool` | Stops live acquisition. |
| `trigger` | `bool` | Dispatches a software trigger for a single frame. |
| `trigger_cam` | `bool` / `dict` | Triggers a frame, optionally accepting an `exposure` override. |
| `exposure` | `int` | Sets the sensor exposure time in microseconds (µs). |
| `gain` | `int` | Sets the hardware sensor gain (0–100). |
| `pixel_clock` | `int` | Sets sensor pixel clock frequency in MHz. |
| `shutter` | `str` | Sets shutter mode (`"Global"`, `"Global alt."`, `"Rolling"`, `"Rolling glob."`). |
| `trigger_mode` | `str` | Selects acquisition mode: `"Video"` or `"Trigger"`. |
| `start_dark_ref`| `bool` | Initiates the dark reference capture sequence. |
| `n_frames` | `int` | Sets the number of frames to average for dark reference. |
| `width`, `height` | `int` | Adjusts sensor AOI dimensions (even pixel values required). |
