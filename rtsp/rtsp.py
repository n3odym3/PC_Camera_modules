import cv2
import urllib.parse
from loguru import logger

class Rtsp:
    def __init__(self) -> None:
        self.cap = None
        self.url = ""
        self.user = ""
        self.password = ""
        self.width = 0
        self.height = 0
        self.cam_name = "RTSP_Stream"

    def build_authenticated_url(self, url: str, user: str, password: str) -> str:
        """
        Takes a base URL and username/password, and builds a fully authenticated RTSP URL.
        If URL already has credentials or username/password are empty, returns the URL formatted with rtsp:// prefix.
        """
        url = url.strip()
        if not url:
            return ""
            
        # Strip existing rtsp:// prefix if present to normalize
        if url.startswith("rtsp://"):
            url_core = url[7:]
        else:
            url_core = url
            
        # If url already has credentials, do not overwrite or append
        if "@" in url_core:
            return f"rtsp://{url_core}"
            
        if not user and not password:
            return f"rtsp://{url_core}"
            
        # Percent-encode username and password to handle special characters
        safe_user = urllib.parse.quote(user, safe="")
        safe_pass = urllib.parse.quote(password, safe="")
        return f"rtsp://{safe_user}:{safe_pass}@{url_core}"

    def init(self, url: str, user: str = "", password: str = "") -> tuple[bool, str | tuple[int, int]]:
        """
        Initialize the RTSP stream.
        Returns a tuple (success, info_or_dimensions).
        """
        self.url = url
        self.user = user
        self.password = password
        
        full_url = self.build_authenticated_url(url, user, password)
        if not full_url:
            return False, "Empty URL"
            
        if self.cap is not None:
            self.cap.release()
            
        logger.info(f"Opening RTSP stream: {url} (user: {user})")
        
        try:
            self.cap = cv2.VideoCapture(full_url)
            
            if not self.cap.isOpened():
                self.cap = None
                return False, "VideoCapture could not open RTSP source"

            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
                
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # If resolution comes back as 0, read one frame to decode dimensions
            if self.width <= 0 or self.height <= 0:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.height, self.width = frame.shape[:2]
                else:
                    self.cap.release()
                    self.cap = None
                    return False, "Could not read initial frame to determine dimensions"
                    
            logger.info(f"RTSP stream initialized at {self.width}x{self.height}")
            return True, (self.width, self.height)
            
        except Exception as e:
            logger.error(f"Error during RTSP initialization: {e}")
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            return False, str(e)

    def grab(self) -> bool:
        """
        Grab the next frame from the stream without decoding it.
        Used in Trigger mode to keep the RTSP buffer fresh with minimal CPU load.
        """
        cap = self.cap
        if cap is not None and cap.isOpened():
            try:
                return bool(cap.grab())
            except Exception as e:
                logger.error(f"Error grabbing frame from RTSP stream: {e}")
        return False

    def read(self):
        """
        Read and decode a single frame from the stream.
        Returns (success, frame)
        """
        cap = self.cap
        if cap is not None and cap.isOpened():
            try:
                ret, frame = cap.read()
                if ret and frame is not None:
                    return True, frame
            except Exception as e:
                logger.error(f"Error reading frame from RTSP stream: {e}")
        return False, None

    def close(self) -> None:
        """Release the capture stream."""
        cap = self.cap
        if cap is not None:
            self.cap = None
            try:
                cap.release()
            except Exception as e:
                logger.error(f"Error releasing RTSP stream: {e}")
            logger.info("RTSP connection closed.")
