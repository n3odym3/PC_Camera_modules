# RTSP Camera Module

Controls Network / IP cameras streaming over **RTSP** (Real-Time Streaming Protocol) via OpenCV and FFmpeg. This module handles URL construction, authentication, live continuous streaming, software-triggered frame capture, and resilient background stream auto-reconnection.

---

## Technical Specifications

- **Accepted Inputs**: `In` (Type: `CMD_DICT` for remote automation and control)
- **Outputs**: `Color Frame (BGR)` (Type: `FRAME`, emits `(np.ndarray, "RTSPCam_<UUID>")` containing 8-bit BGR color image arrays)
- **Transport**: RTSP over TCP/UDP with optional percent-encoded credential injection.

---

## Key Features & Controls

- **Authentication & Stream URL**:
  - Input stream IP/port/path (e.g. `192.168.1.100:554/stream1`).
  - Optional username and password credentials with dynamic URL encoding and password masking toggle.
- **Acquisition Modes**:
  - `Continuous`: Real-time live frame capture and emission to downstream modules.
  - `Trigger`: Holds grabbed frames in buffer and only dispatches the latest frame when manually triggered or requested via automation.
- **Resilient Auto-Reconnection**:
  - Background worker thread detects read drops and network disconnections.
  - Automatically attempts reconnection with retry backoff to resume streaming without restarting the application.
- **Image Rotation**:
  - Supports image orientation via `config.Cameras.RTSP.rotation` (0°, 90°, 180°, 270°).

---

## Remote Control API (`CMD_DICT`)

The module can be controlled remotely by sending a dictionary command to its input:

| Command Key | Type | Description |
| :--- | :--- | :--- |
| `connect` | `bool` | Connects to the RTSP camera stream. |
| `disconnect` | `bool` | Disconnects and stops background acquisition. |
| `trigger` | `bool` | Requests dispatch of the next grabbed frame. |
| `trigger_cam` | `bool` | Requests dispatch of the next grabbed frame. |
| `mode` | `str` | Selects acquisition mode: `"Continuous"` or `"Trigger"`. |
| `url` | `str` | Updates the RTSP stream URL. |
| `user` | `str` | Updates the authentication username. |
| `password` | `str` | Updates the authentication password. |
