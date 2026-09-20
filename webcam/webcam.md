# Webcam Module

Controls **generic USB webcams and DirectShow video grabbers** via OpenCV's capture interface and DirectShow stream capabilities. It is ideal for rapid testing, demo setups, machine vision inspection, and color (BGR) video acquisition.

---

## Technical Specifications

- **Accepted Inputs**: `In` (Type: `CMD_DICT` for automation and programmatic control)
- **Outputs**:
  - `Color Frame (BGR)` (Type: `FRAME`, emits `(np.ndarray, "Webcam_<DeviceName>")` with 8-bit BGR image array)

---

## Key Features & Controls

- **Lazy Hardware Initialization**:
  - Instantiating the module does not immediately capture or lock any camera device.
  - Available video grabbers are discovered and listed in the UI; acquisition only starts when a camera is chosen or when **Start** is pressed.
- **Auto-Discovery & DirectShow Enumeration**:
  - Click **Refresh** to query all connected USB video grabber devices.
  - Select any available device from the combo box to initialize and probe its features.
- **Dynamic Resolution & Capability Scanning**:
  - Uses DirectShow stream capabilities (`FilterGraph`) to probe all hardware-supported video formats, resolutions, and theoretical maximum framerates.
  - Labels display exact dimensions and limits (e.g. `1920x1080 (max 60 FPS)`, `1280x720 (max 60 FPS)`, `640x480 (max 90 FPS)`).
  - Automatic fallback to standard formats when capability querying is unsupported.
- **Theoretical & Live FPS Monitoring**:
  - Displays the driver's theoretical upper bound alongside real-time measured acquisition FPS.
- **Acquisition Modes**:
  - `Continuous`: Real-time streaming with live FPS counter.
  - `Trigger`: On-demand capture via the `Trigger` UI button or external `CMD_DICT` signal.
- **UVC / Hardware Controls**:
  - Dedicated **Open Camera Settings** button triggers the native DirectShow device properties dialog (`CAP_PROP_SETTINGS`) for real-time sensor parameter tuning (exposure, white balance, gain, focus).
- **Image Orientation**:
  - Supports image rotation via `config.Cameras.Webcam.rotation` (0°, 90°, 180°, 270°).

---

## Remote Control API (`CMD_DICT`)

The module accepts dictionary commands via its input for external pipeline automation:

| Command Key | Type | Description |
| :--- | :--- | :--- |
| `start` | `bool` | Starts the video acquisition thread. |
| `stop` | `bool` | Stops the video acquisition thread. |
| `trigger` | `bool` | In Trigger mode, requests capture and dispatch of a single frame. |
| `trigger_cam` | `bool` | Alias for `trigger`. |
| `mode` | `str` | Sets acquisition mode: `"Continuous"` or `"Trigger"`. |
| `camera` | `str` | Selects and connects to a camera device by name. |
| `select_cam` | `str` | Alias for `camera`. |
| `resolution` | `str` | Changes stream resolution (e.g. `"1920x1080"` or `"1920x1080 (max 60 FPS)"`). |
| `open_settings` | `bool` | Opens the native DirectShow / UVC device settings dialog. |
