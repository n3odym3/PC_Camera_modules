# Virtual Camera Module

Streams image frames from `Pipeline_creator` pipelines into an OS-level virtual camera device via `pyvirtualcam`. The resulting virtual camera can be selected as a video input source in third-party software such as **OBS Studio**, **Microsoft Teams**, **Zoom**, **Discord**, **Google Meet**, or web browsers.

---

## Technical Specifications

- **Accepted Inputs**:
  - `In` (Type: `FRAME` for image streaming, or `CMD_DICT` for remote automation)
- **Outputs**:
  - `Color Frame (BGR)` (Type: `FRAME`, transparent passthrough of the incoming image frame for pipeline chaining)
  - `Status` (Type: `STATUS_DICT`, emits `{is_running, device, fps, frames_sent}`)

---

## Key Features & Controls

- **Multi-Backend Architecture**:
  - `obs`: Native OBS Studio DirectShow virtual camera driver (recommended on Windows).
  - `unitycapture`: DirectShow Unity Capture virtual video device.
  - `v4l2loopback`: Linux Video4Linux loopback virtual camera device.
  - `auto`: Automatically detects and initializes the best available driver.
- **Color & Depth Normalization**:
  - Automatically handles 1-channel Grayscale, 3-channel BGR/RGB, and 4-channel BGRA/RGBA images.
  - Automatically scales non-8-bit frames (e.g. 12-bit, 16-bit) to standard 8-bit depth using min-max normalization.
  - Ensures C-contiguous memory layout before transmission to prevent buffer format exceptions.
- **Resolution & Aspect Ratio Management**:
  - **Auto-fit**: Automatically detects the incoming frame dimensions and sets the virtual camera resolution accordingly.
  - **Manual Dimensions**: Configure target width, height, and target framerate (FPS). Frames are automatically resized to match the active virtual camera dimensions.
- **Live Telemetry & Diagnostics**:
  - Real-time display of active device name, total frames sent, and live transmitted FPS.
  - Instant status feedback (Running, Stopped, Error) and clear driver warning messages if `pyvirtualcam` is missing.
- **Pipeline Chaining (Passthrough)**:
  - Forwards the unmodified frame downstream to `Color Frame (BGR)`, allowing users to insert the Virtual Camera module anywhere in the processing chain without breaking the flow.

---

## Prerequisites

1. **Python Package**:
   ```bash
   pip install pyvirtualcam
   ```

2. **Virtual Camera Driver**:
   - **OBS Studio (Recommended)**: Installing [OBS Studio](https://obsproject.com/) automatically registers the Windows DirectShow virtual camera driver.
   - **Unity Capture**: Alternative DirectShow virtual capture filter.

---

## Remote Control API (`CMD_DICT`)

The module can be automated by sending dictionary commands to its input:

| Command Key | Type | Description |
| :--- | :--- | :--- |
| `start` | `bool` | Starts the virtual camera output stream. |
| `stop` | `bool` | Stops the virtual camera output stream. |
| `backend` | `str` | Sets driver backend (`"obs"`, `"unitycapture"`, `"v4l2loopback"`, `"auto"`). |
| `width` | `int` | Sets the virtual camera target width in pixels (min 160). |
| `height` | `int` | Sets the virtual camera target height in pixels (min 120). |
| `fps` | `int` | Sets the target output framerate (1–120 FPS). |
| `auto_fit` | `bool` | Enables or disables auto-fitting resolution to incoming frame sizes. |
