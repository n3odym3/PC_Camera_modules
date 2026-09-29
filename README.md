# Camera Modules for Pipeline Creator

A collection of camera acquisition and streaming modules for [**Pipeline Creator**](https://github.com/n3odym3/Pipeline_creator).

This repository contains camera modules (USB webcams, industrial cameras, RTSP streams, and virtual camera output for third-party software) designed for plug-and-play integration into the Pipeline Creator ecosystem.

---

## Installation

To install and use one or more modules in **Pipeline Creator**:

1. **Install Module Dependencies**:

   Install the required Python packages and SDKs for the module(s) you intend to use (e.g. OpenCV, NumPy, VmbPy for Alvium, IDS peak SDK, Hamamatsu DCAM-API / pyDCAM, etc.) within your Pipeline Creator environment:
   ```bash
   pip install opencv-python numpy
   # Install any hardware-specific SDKs/packages as needed (e.g., vmbpy, ids-peak, pyDCAM, etc.)
   ```

2. **Copy Modules to Pipeline Creator**:
   
   Copy and paste the folder(s) of the desired module(s) (such as `webcam`, `Alvium`, `Hamamatsu`, `ids`, `rtsp`, etc.) directly into the `modules/` folder of your **Pipeline Creator** installation:
   ```
   Pipeline_creator/
   ├── modules/
   │   ├── webcam/
   │   ├── Alvium/
   │   ├── Hamamatsu/
   │   ├── ids/
   │   ├── rtsp/
   │   └── ...
   ```

3. **Launch Pipeline Creator**:

   The copied modules will be automatically detected and made available in the Pipeline Creator node editor.

---

## Available Modules

| Module | Description | Key Dependencies |
| :--- | :--- | :--- |
| [**webcam**](./webcam/) | Acquisition for generic USB webcams and DirectShow grabbers via OpenCV. Supports continuous and trigger modes, resolution scanning, and camera property dialogs. | `opencv-python`, `numpy` |
| [**Alvium**](./Alvium/) | Support for Allied Vision Alvium industrial cameras via the Vimba / VmbPy SDK. | `vmbpy`, Vimba / Vimba X SDK |
| [**Hamamatsu**](./Hamamatsu/) | Support for Hamamatsu scientific CMOS (sCMOS) cameras via DCAM-API and `pyDCAM`. Supports continuous streaming, trigger modes (hardware & software), real-time dark reference correction, hardware binning, readout speed control, and native 16-bit monochrome output. | Hamamatsu DCAM-API, `pyDCAM`, `numpy`, `opencv-python`, `loguru` |
| [**ids**](./ids/) | Support for IDS Imaging industrial cameras. | IDS peak / uEye SDK, `numpy` |
| [**rtsp**](./rtsp/) | Streaming input for RTSP network feeds (IP cameras, streaming servers). | `opencv-python` |
| [**seecam_win**](./seecam_win/) | Windows support for e-con Systems See3CAM USB cameras. | DirectShow |
| [**virtual_camera**](./virtual_camera/) | Streams pipeline frames into an OS-level virtual camera device (OBS Virtualcam, UnityCapture, v4l2loopback) so they can be consumed as a camera source by third-party software (OBS Studio, Zoom, Teams, Discord, web browsers, etc.). | `pyvirtualcam`, `numpy`, virtual camera driver (e.g. OBS Studio) |

---