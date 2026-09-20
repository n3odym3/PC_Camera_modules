# SeeCam Camera Module

Controls **e-con Systems See3CAM** series industrial cameras via DirectShow, including **See3CAM_20CUG (CU20G)** and **See3CAM_CU55M (CU55M)**. This module captures and streams dual-mode outputs, delivering standard 8-bit RGB/BGR frames as well as high-depth raw grayscale frames (Y8, Y12, Y16).

---

## Technical Specifications

- **Accepted Inputs**: `In` (Type: `CMD_DICT` for remote automation and control)
- **Outputs**:
  - `8bit RGB` (Type: `FRAME`, emits `(np.ndarray, "Seecam_<Name>")` with 8-bit BGR image array)
  - `16bit gray` (Type: `FRAME16`, emits `(np.ndarray, "Seecam_raw_<Name>")` with raw 12-bit / 16-bit / 8-bit image array)

---

## Key Features & Controls

- **Supported Models**:
  - **See3CAM_20CUG (CU20G)**: 2MP global shutter monochrome camera (OV2311). Supports **Y8** (8-bit) and **Y16** (10-bit raw in 16-bit word).
  - **See3CAM_CU55M (CU55M)**: 5MP monochrome camera (AR0521). Supports **Y8** (8-bit) and **Y12** (packed 12-bit, unpacked to 16-bit [0..4095]).
  - **Generic See3CAM**: Dynamically discovers all stream capabilities via DirectShow.
- **Resolution & Pixel Format Selection**:
  - Lists matching stream capabilities with theoretical frame rates (e.g. `2592x1944 @ 30 FPS (Y12)` or `1920x1080 @ 60 FPS (Y16)`).
  - Seamless on-the-fly stream reconfiguration.
- **Adaptive 8-bit Conversion**:
  - `Auto`: Automatically adapts bit-shifting based on the detected camera model and active format:
    - `Y8`: Pass-through.
    - `CU55M Y12`: Bit-shift `>> 4` (12-bit to 8-bit).
    - `CU20G Y16`: Bit-shift `>> 2` (10-bit raw in 16-bit container to 8-bit).
  - `Normalize`: Dynamic min-max contrast stretching.
  - Manual shifts: `8-bit`, `10-bit`, `12-bit`, `16-bit`.
- **Acquisition Modes**:
  - `Continuous`: Real-time streaming with live FPS counter.
  - `Trigger`: On-demand capture via the `Trigger Frame` UI button or external `CMD_DICT` trigger input.
- **Hardware & UVC Controls**:
  - Gain and Exposure sliders with dynamic ranges queried directly from the device.
  - Auto-exposure loop with live peak intensity feedback.
- **Image Orientation**:
  - Configurable rotation via `config.Cameras.Seecam.rotation` (0°, 90°, 180°, 270°).

---

## Remote Control API (`CMD_DICT`)

The module can be automated by sending dictionary commands to its input:

| Command Key | Type | Description |
| :--- | :--- | :--- |
| `start` | `bool` | Starts the acquisition worker thread. |
| `stop` | `bool` | Stops the acquisition worker thread. |
| `trigger` | `bool` | Triggers acquisition and dispatch of a single frame. |
| `trigger_cam` | `bool` | Triggers acquisition and dispatch of a single frame. |
| `mode` | `str` | Selects acquisition mode: `"Continuous"` or `"Trigger"`. |
| `gain` | `int` | Sets the hardware gain value. |
| `exposure` | `int` | Sets the hardware exposure value. |
| `auto_exposure` | `bool` | Enables or disables the auto-exposure feedback loop. |
| `depth_mode` | `str` | Sets bit-depth conversion mode (`"Auto"`, `"Normalize"`, `"8-bit"`, etc.). |
| `resolution` | `str` | Switches active resolution/format to a matching capability label. |
| `camera` | `str` | Selects and initializes a connected camera by device name. |
