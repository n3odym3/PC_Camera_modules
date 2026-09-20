from loguru import logger
import numpy as np
import time
import cv2 

from comtypes import COMError, CoInitialize, GUID
from ctypes import c_long, byref

from .pygrabber.dshow_graph import FilterGraph, FilterType
from .pygrabber.dshow_ids import MediaSubtypes, MediaTypes, FormatTypes
from .pygrabber.dshow_core import IAMCameraControlIID, IAMVideoProcAmpIID, IAMStreamConfig, qedit

CameraControl_Pan      = 0
CameraControl_Tilt     = 1
CameraControl_Roll     = 2
CameraControl_Zoom     = 3
CameraControl_Exposure = 4
CameraControl_Iris     = 5
CameraControl_Focus    = 6
CameraControl_Flags_Auto   = 0x0001
CameraControl_Flags_Manual = 0x0002

VideoProcAmp_Brightness            = 0
VideoProcAmp_Contrast              = 1
VideoProcAmp_Hue                   = 2
VideoProcAmp_Saturation            = 3
VideoProcAmp_Sharpness             = 4
VideoProcAmp_Gamma                 = 5
VideoProcAmp_ColorEnable           = 6
VideoProcAmp_WhiteBalance          = 7
VideoProcAmp_BacklightCompensation = 8
VideoProcAmp_Gain                  = 9
VideoProcAmp_Flags_Auto   = 0x0001
VideoProcAmp_Flags_Manual = 0x0002


class Seecam:
	"""
	DirectShow acquisition and control backend for e-con Systems See3CAM cameras.
	Supports CU20G (OV2311 monochrome global shutter), CU55M (AR0521 monochrome 5MP),
	and generic See3CAM cameras with Y8, Y12, and Y16 formats.
	"""

	def __init__(self):
		try:
			CoInitialize()
		except OSError:
			pass

		self.defined = False
		self.graph = None
		self.last_frame = None
		self.frame_ready = False
		self.camera_filter = None
		self.cam_ctrl = None
		self.proc_amp = None
		self.gain_prop_id = VideoProcAmp_Gain

		self.cam_index = -1
		self.cam_name = ""
		self.model = "Generic"
		self.model_desc = "See3CAM"

		self.format = ""
		self.width = 0
		self.height = 0
		self.fps = 0.0
		self.active_format_index = 0
		self.available_formats = []

		self.max_exposure_check = 4
		self.exposure_check_count = 0

	def list_devices(self):
		"""Returns list of video input devices via DirectShow."""
		try:
			temp_graph = FilterGraph()
			devices = temp_graph.get_input_devices()
			temp_graph.stop()
			return devices
		except Exception as e:
			logger.warning(f"Error enumerating DirectShow devices: {e}")
			return []

	def _detect_model(self, cam_name: str):
		"""Detects camera model characteristics (CU20G vs CU55M)."""
		name_lower = str(cam_name).lower()
		if "20cug" in name_lower or "cu20" in name_lower:
			self.model = "CU20G"
			self.model_desc = "CU20G (Global Shutter 2MP OV2311)"
		elif "55m" in name_lower or "cu55" in name_lower:
			self.model = "CU55M"
			self.model_desc = "CU55M (Monochrome 5MP AR0521)"
		else:
			self.model = "See3CAM"
			self.model_desc = f"See3CAM ({cam_name})"

	def init(self, cam, format_index=None):
		"""
		Initializes the camera graph with the specified camera and format index.
		"""
		self.close()

		camlist = self.list_devices()
		if isinstance(cam, str):
			if cam in camlist:
				cam_idx = camlist.index(cam)
				cam_name = cam
			else:
				matched = [i for i, name in enumerate(camlist) if cam.lower() in name.lower()]
				if matched:
					cam_idx = matched[0]
					cam_name = camlist[cam_idx]
				else:
					logger.error(f"Device '{cam}' not found in {camlist}")
					return False, None
		else:
			cam_idx = int(cam)
			cam_name = camlist[cam_idx] if 0 <= cam_idx < len(camlist) else f"Device_{cam_idx}"

		self.cam_index = cam_idx
		self.cam_name = cam_name
		self._detect_model(cam_name)

		try:
			self.graph = FilterGraph()
			self.graph.add_video_input_device(self.cam_index)
			video_input = self.graph.get_input_device()

			# Enumerate all supported stream capabilities
			self.available_formats = self.get_available_formats()

			# Determine format index to configure
			if format_index is None:
				format_index = self._pick_default_format_index()

			if format_index is not None and self.available_formats:
				valid_indices = [f['index'] for f in self.available_formats]
				if format_index in valid_indices:
					video_input.set_format(format_index)
					self.active_format_index = format_index
					for f in self.available_formats:
						if f['index'] == format_index:
							self.fps = f['fps']
							break

			self.graph.add_null_render()
			self.graph.add_sample_grabber(self.frame_callback)
			sample_grabber = self.graph.filters[FilterType.sample_grabber]

			self.format, self.width, self.height = video_input.get_current_format()

			# Prepare candidate subtypes for SampleGrabber
			subtypes_to_try = []
			if self.format == "Y8":
				subtypes_to_try = [
					'{20203859-0000-0010-8000-00AA00389B71}',  # Y8
					'{30303859-0000-0010-8000-00AA00389B71}',  # Y800
					'{59455247-0000-0010-8000-00AA00389B71}',  # GREY
					'{E436EB7A-524F-11CE-9F53-0020AF0BA770}',  # RGB8
				]
			elif self.format == "Y12":
				subtypes_to_try = [MediaSubtypes.Y12]
			elif self.format == "Y16":
				subtypes_to_try = [MediaSubtypes.Y16]
			elif self.format == "RGB24":
				subtypes_to_try = [MediaSubtypes.RGB24]
			elif self.format == "UYVY":
				subtypes_to_try = ['{59565955-0000-0010-8000-00AA00389B71}']
			elif self.format == "YUY2":
				subtypes_to_try = ['{32595559-0000-0010-8000-00AA00389B71}']
			else:
				subtypes_to_try = [MediaSubtypes.RGB24]

			connected = False
			for st_guid in subtypes_to_try:
				# Strategy A: Subtype + FORMAT_VideoInfo (Standard DirectShow requirement for Y8/Y800)
				try:
					sg_type = qedit._AMMediaType()
					sg_type.majortype = GUID(MediaTypes.Video)
					sg_type.subtype = GUID(st_guid)
					sg_type.formattype = GUID(FormatTypes.FORMAT_VideoInfo)
					sample_grabber.sample_grabber.SetMediaType(sg_type)
					self._connect_preview_graph()
					connected = True
					logger.debug(f"Preview connected with subtype {st_guid} (FORMAT_VideoInfo)")
					break
				except Exception as e_a:
					logger.debug(f"Subtype {st_guid} + FORMAT_VideoInfo failed: {e_a}")

				# Strategy B: Subtype without formattype
				try:
					sample_grabber.set_media_type(MediaTypes.Video, st_guid)
					self._connect_preview_graph()
					connected = True
					logger.debug(f"Preview connected with subtype {st_guid} (unconstrained formattype)")
					break
				except Exception as e_b:
					logger.debug(f"Subtype {st_guid} unconstrained failed: {e_b}")

			# Strategy C: Unconstrained subtype (generic video)
			if not connected:
				try:
					sg_type = qedit._AMMediaType()
					sg_type.majortype = GUID(MediaTypes.Video)
					sample_grabber.sample_grabber.SetMediaType(sg_type)
					self._connect_preview_graph()
					connected = True
					logger.debug("Preview connected with generic video type")
				except Exception as e_c:
					logger.debug(f"Generic video type connection failed: {e_c}")

			if not connected:
				logger.error(f"All preview connect strategies failed on format {self.format}")
				raise COMError(-2147220969, "VFW_E_CANNOT_CONNECT", (None, None, None, 0, None))

			self.graph.run()

			self.camera_filter = self.graph.filters[FilterType.video_input].instance
			self.cam_ctrl = self.camera_filter.QueryInterface(IAMCameraControlIID)
			self.proc_amp = self.camera_filter.QueryInterface(IAMVideoProcAmpIID)

			proc_params = self.list_video_proc_parameters()
			if proc_params.get("Gain") is not None:
				self.gain_prop_id = VideoProcAmp_Gain
			else:
				self.gain_prop_id = VideoProcAmp_Brightness

			self.defined = True
			logger.success(f"Seecam {self.model_desc} initialized: {self.format} {self.width}x{self.height} @ ~{self.fps:.0f} FPS")
			return True, (self.format, self.width, self.height, self.fps, self.model_desc)

		except Exception as e:
			logger.error(f"Seecam init error on '{cam_name}': {e}")
			self.close()
			return False, None

	def _connect_preview_graph(self):
		"""Connects video_input -> sample_grabber -> null_renderer safely with pin cleanups and fallback."""
		video_out = self.graph.filters[FilterType.video_input].get_out()
		sg_in = self.graph.filters[FilterType.sample_grabber].get_in()
		sg_out = self.graph.filters[FilterType.sample_grabber].get_out()
		render_in = self.graph.filters[FilterType.render].get_in()

		# Ensure pins are clean before connecting
		for pin in [video_out, sg_in, sg_out, render_in]:
			try:
				self.graph.graph_builder.Disconnect(pin)
			except Exception:
				pass

		# Connect Step 1: video_out -> sg_in
		try:
			self.graph.graph_builder.ConnectDirect(video_out, sg_in, None)
		except Exception as e_vin_d:
			try:
				self.graph.graph_builder.Connect(video_out, sg_in)
			except Exception as e_vin_c:
				logger.error(f"[PIN ERROR] video_out -> sg_in connection failed: {e_vin_c}")
				raise e_vin_c

		# Connect Step 2: sg_out -> render_in (with fallback to Render if NullRender rejects format)
		try:
			self.graph.graph_builder.ConnectDirect(sg_out, render_in, None)
		except Exception as e_rout_d:
			try:
				self.graph.graph_builder.Connect(sg_out, render_in)
			except Exception as e_rout_c:
				logger.warning(f"Connect sg_out -> render_in failed ({e_rout_c}), attempting intelligent Render(sg_out)...")
				try:
					self.graph.graph_builder.Render(sg_out)
				except Exception as e_render:
					logger.error(f"[PIN ERROR] Render(sg_out) failed: {e_render}")
					raise e_rout_c

		self.graph.filters[FilterType.sample_grabber].initialize_after_connection()
		self.graph.is_recording = False

	def _pick_default_format_index(self):
		"""Picks an optimal format based on camera model and capabilities."""
		if not self.available_formats:
			return 0

		preferred_order = ["Y16", "Y12", "Y8"] if self.model == "CU20G" else ["Y12", "Y16", "Y8"]

		for target_fmt in preferred_order:
			matching = [f for f in self.available_formats if f['media_type'] == target_fmt]
			if matching:
				matching.sort(key=lambda x: (x['width'] * x['height'], x['fps']), reverse=True)
				return matching[0]['index']

		return self.available_formats[0]['index']

	def get_available_formats(self):
		"""
		Scans and returns all hardware-supported stream capabilities with metadata.
		"""
		if not self.graph:
			return []

		try:
			raw_formats = self.graph.get_input_device().get_formats()
			parsed = []
			seen = set()

			for item in raw_formats:
				idx = int(item.get('index', 0))
				w = int(item.get('width', 0))
				h = int(item.get('height', 0))
				if w <= 0 or h <= 0:
					continue

				fmt_str = str(item.get('media_type_str', ''))
				min_fps = float(item.get('min_framerate', 0.0))
				max_fps = float(item.get('max_framerate', 0.0))
				fps = max(min_fps, max_fps)

				key = (w, h, fmt_str, round(fps, 1))
				if key not in seen:
					seen.add(key)
					parsed.append({
						'index': idx,
						'width': w,
						'height': h,
						'media_type': fmt_str,
						'fps': fps,
						'label': f"{w}x{h} @ {fps:.0f} FPS ({fmt_str})" if fmt_str else f"{w}x{h} @ {fps:.0f} FPS",
						'short_label': f"{w}x{h} ({fmt_str})"
					})

			parsed.sort(key=lambda x: (x['width'] * x['height'], x['fps']), reverse=True)
			return parsed
		except Exception as e:
			logger.warning(f"Error scanning formats: {e}")
			return []

	def set_format_by_index(self, format_index: int) -> bool:
		"""
		Reconfigures the DirectShow stream to a different format and resolution.
		"""
		if self.cam_index < 0:
			return False
		ret, _ = self.init(self.cam_index, format_index=format_index)
		return ret

	def frame_callback(self, buffer, blen):
		"""
		Callback invoked by DirectShow SampleGrabber filter for each incoming frame.
		"""
		try:
			buffer_np = np.ctypeslib.as_array(buffer, shape=(blen,))

			if self.format == "Y8":
				expected_bytes = self.width * self.height
				if blen >= expected_bytes:
					frame = buffer_np[:expected_bytes].view(dtype=np.uint8).reshape((self.height, self.width))
				else:
					return

			elif self.format == "Y12":
				# 12-bit packed: 2 pixels stored in 3 bytes
				expected_bytes = (self.width * self.height * 3) // 2
				if blen >= expected_bytes:
					triplets = buffer_np[:expected_bytes].reshape(-1, 3)
					p1 = ((triplets[:, 0].astype(np.uint16) << 4) | (triplets[:, 2] & 0x0F))
					p2 = ((triplets[:, 1].astype(np.uint16) << 4) | ((triplets[:, 2] >> 4) & 0x0F))

					pixels = np.empty((self.width * self.height,), dtype=np.uint16)
					pixels[0::2] = p1
					pixels[1::2] = p2
					frame = pixels.reshape((self.height, self.width))
				else:
					return

			elif self.format == "Y16":
				expected_bytes = self.width * self.height * 2
				if blen >= expected_bytes:
					frame = buffer_np[:expected_bytes].view(dtype=np.uint16).reshape((self.height, self.width))
				else:
					return

			elif self.format in ["UYVY", "YUY2"]:
				expected_bytes = self.width * self.height * 2
				if blen >= expected_bytes:
					raw = buffer_np[:expected_bytes].reshape((self.height, self.width, 2))
					code = cv2.COLOR_YUV2BGR_UYVY if self.format == "UYVY" else cv2.COLOR_YUV2BGR_YUY2
					bgr = cv2.cvtColor(raw, code)
					frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
				else:
					return
			else:
				frame = None

			if frame is not None:
				self.last_frame = frame
				self.frame_ready = True

		except Exception:
			pass

	def read(self, instant=False):
		"""Reads the latest grabbed frame."""
		frame = None
		self.frame_ready = False
		if self.defined and self.graph:
			try:
				if self.graph.grab_frame():
					if not instant:
						timeout = time.time() + 0.5
						while not self.frame_ready and time.time() < timeout:
							time.sleep(0.001)
					frame = self.last_frame
				return True, frame
			except Exception:
				return False, None
		return False, frame

	def convert_to_8bit(self, frame: np.ndarray, depth_mode: str = "Auto") -> np.ndarray:
		"""
		Converts high-depth monochrome frame (Y12, Y16) to 8-bit.
		In 'Auto' mode:
		  - Y8: pass-through
		  - Y12 (CU55M): >> 4
		  - Y16 (CU20G): >> 2 (10-bit raw in 16-bit word)
		  - Y16 generic: >> 8 or >> 2 depending on dynamic range
		"""
		if frame is None:
			return None

		if depth_mode == "Auto":
			if self.format == "Y8" or frame.dtype == np.uint8:
				return frame
			elif self.format == "Y12":
				return (frame >> 4).astype(np.uint8)
			elif self.format == "Y16":
				if self.model == "CU20G":
					return (frame >> 2).astype(np.uint8)
				else:
					sample = frame[::8, ::8] if frame.shape[0] >= 8 else frame
					if np.max(sample) < 1024:
						return (frame >> 2).astype(np.uint8)
					return (frame >> 8).astype(np.uint8)
			else:
				if frame.dtype == np.uint8:
					return frame
				return (frame >> 8).astype(np.uint8)

		elif depth_mode == "Normalize":
			return cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
		elif depth_mode == "8-bit":
			return frame.astype(np.uint8)
		elif depth_mode == "10-bit":
			return (frame >> 2).astype(np.uint8)
		elif depth_mode == "12-bit":
			return (frame >> 4).astype(np.uint8)
		elif depth_mode == "16-bit":
			return (frame >> 8).astype(np.uint8)

		return frame.astype(np.uint8)

	def convert_to_BGR(self, frame: np.ndarray) -> np.ndarray:
		"""Converts 8-bit grayscale frame to 3-channel BGR."""
		if frame is None:
			return None
		if frame.ndim == 2:
			return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
		return frame

	def calc_exposure(self, frame: np.ndarray) -> float:
		"""Calculates maximum pixel intensity for exposure feedback."""
		if frame is None or frame.size == 0:
			return 0.0
		return float(np.max(frame))

	def auto_expos(self, frame, target_exposure=220, threshold=20, min_exposure=-13, max_exposure=0):
		"""Simple auto-exposure feedback loop."""
		frame_exposure = self.calc_exposure(frame)
		current_exposure = self.get_exposure()
		if current_exposure is None:
			return 0

		limits = self.get_exposure_limits()
		if limits:
			min_exposure, max_exposure = limits[0], limits[1]

		if frame_exposure < target_exposure - threshold:
			if self.exposure_check_count > self.max_exposure_check:
				if current_exposure < max_exposure:
					self.set_exposure(current_exposure + 1)
					self.exposure_check_count = 0
			else:
				self.exposure_check_count += 1

		elif frame_exposure > target_exposure + threshold:
			if self.exposure_check_count > self.max_exposure_check:
				if current_exposure > min_exposure:
					self.set_exposure(current_exposure - 1)
					self.exposure_check_count = 0
			else:
				self.exposure_check_count += 1

		return self.get_exposure()

	def list_camera_parameters(self):
		if not self.cam_ctrl:
			return {}
		
		camera_control_props = [
			(CameraControl_Pan,      "Pan"),
			(CameraControl_Tilt,     "Tilt"),
			(CameraControl_Roll,     "Roll"),
			(CameraControl_Zoom,     "Zoom"),
			(CameraControl_Exposure, "Exposure"),
			(CameraControl_Iris,     "Iris"),
			(CameraControl_Focus,    "Focus"),
		]

		result = {}
		for prop_id, prop_name in camera_control_props:
			min_val = c_long()
			max_val = c_long()
			step_val = c_long()
			default_val = c_long()
			caps_flags = c_long()

			try:
				hr = self.cam_ctrl.GetRange(
					prop_id,
					byref(min_val),
					byref(max_val),
					byref(step_val),
					byref(default_val),
					byref(caps_flags)
				)
				if hr == 0:
					result[prop_name] = {
						"min": min_val.value,
						"max": max_val.value,
						"step": step_val.value,
						"default": default_val.value,
						"caps": caps_flags.value
					}
				else:
					result[prop_name] = None
			except COMError:
				result[prop_name] = None

		return result

	def get_exposure_limits(self):
		cam_params = self.list_camera_parameters()
		if cam_params.get("Exposure") is not None:
			return cam_params["Exposure"]["min"], cam_params["Exposure"]["max"], cam_params["Exposure"].get("step", 1)
		return None

	def set_exposure(self, exposure_value: int, manual: bool = True) -> bool:
		if not self.cam_ctrl:
			return False

		flags = CameraControl_Flags_Manual if manual else CameraControl_Flags_Auto
		try:
			hr = self.cam_ctrl.Set(CameraControl_Exposure, int(exposure_value), flags)
			return hr == 0
		except COMError:
			return False

	def get_exposure(self):
		if not self.cam_ctrl:
			return None
		value = c_long()
		flags = c_long()
		try:
			hr = self.cam_ctrl.Get(CameraControl_Exposure, byref(value), byref(flags))
			if hr == 0:
				return value.value
		except COMError:
			return None
		return None

	def list_video_proc_parameters(self):
		if not self.proc_amp:
			return {}

		video_proc_amp_props = [
			(VideoProcAmp_Brightness,            "Brightness"),
			(VideoProcAmp_Contrast,              "Contrast"),
			(VideoProcAmp_Hue,                   "Hue"),
			(VideoProcAmp_Saturation,            "Saturation"),
			(VideoProcAmp_Sharpness,             "Sharpness"),
			(VideoProcAmp_Gamma,                 "Gamma"),
			(VideoProcAmp_ColorEnable,           "ColorEnable"),
			(VideoProcAmp_WhiteBalance,          "WhiteBalance"),
			(VideoProcAmp_BacklightCompensation, "BacklightComp"),
			(VideoProcAmp_Gain,                  "Gain"),
		]

		result = {}
		for prop_id, prop_name in video_proc_amp_props:
			min_val = c_long()
			max_val = c_long()
			step_val = c_long()
			default_val = c_long()
			caps_val = c_long()

			try:
				hr = self.proc_amp.GetRange(
					prop_id,
					byref(min_val),
					byref(max_val),
					byref(step_val),
					byref(default_val),
					byref(caps_val)
				)
				if hr == 0:
					result[prop_name] = {
						"min":     min_val.value,
						"max":     max_val.value,
						"step":    step_val.value,
						"default": default_val.value,
						"caps":    caps_val.value,
					}
				else:
					result[prop_name] = None
			except COMError:
				result[prop_name] = None

		return result

	def get_gain_limits(self):
		proc_params = self.list_video_proc_parameters()
		if proc_params.get("Gain") is not None:
			self.gain_prop_id = VideoProcAmp_Gain
			return proc_params["Gain"]["min"], proc_params["Gain"]["max"], proc_params["Gain"].get("step", 1)
		elif proc_params.get("Brightness") is not None:
			self.gain_prop_id = VideoProcAmp_Brightness
			return proc_params["Brightness"]["min"], proc_params["Brightness"]["max"], proc_params["Brightness"].get("step", 1)
		return None

	def set_gain(self, gain_value: int) -> bool:
		if not self.proc_amp:
			return False
		flags = VideoProcAmp_Flags_Manual
		try:
			hr = self.proc_amp.Set(self.gain_prop_id, int(gain_value), flags)
			return hr == 0
		except COMError:
			return False

	def get_gain(self):
		if not self.proc_amp:
			return None
		value = c_long()
		flags = c_long()
		try:
			hr = self.proc_amp.Get(self.gain_prop_id, byref(value), byref(flags))
			if hr == 0:
				return value.value
		except COMError:
			return None
		return None

	def close(self):
		"""Stops the DirectShow graph, disconnects pins, removes filters, and releases COM objects."""
		self.defined = False
		self.frame_ready = False
		self.last_frame = None

		self.cam_ctrl = None
		self.proc_amp = None
		self.camera_filter = None

		if self.graph is not None:
			try:
				if hasattr(self.graph, 'media_control') and self.graph.media_control:
					self.graph.media_control.Stop()
					try:
						state = c_long()
						self.graph.media_control.GetState(500, byref(state))
					except Exception:
						pass
			except Exception:
				pass

			try:
				if hasattr(self.graph, 'filter_graph') and self.graph.filter_graph:
					fg = self.graph.filter_graph
					if hasattr(self.graph, 'filters') and self.graph.filters:
						for ftype, flt in list(self.graph.filters.items()):
							try:
								if hasattr(flt, 'pins'):
									for pin in flt.pins:
										try:
											self.graph.graph_builder.Disconnect(pin)
										except Exception:
											pass
								if hasattr(flt, 'instance') and flt.instance:
									fg.RemoveFilter(flt.instance)
							except Exception:
								pass
						self.graph.filters.clear()
			except Exception:
				pass

			self.graph = None

		import gc
		gc.collect()
		time.sleep(0.05)
