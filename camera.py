"""Per-session webcam processor. No Streamlit calls from the WebRTC worker."""
import threading
import time

import av
import cv2
from streamlit_webrtc import VideoProcessorBase

from model import FRAMES
from vision import make_tracker, process_frame


class SignCamera(VideoProcessorBase):
    def __init__(self, predictor):
        self.predictor = predictor
        self.lock = threading.Lock()
        self.tracker = None
        self.sequence = []
        self.tracked = 0
        self.start_at = None
        self.last_frame_at = 0
        self.ranked = []
        self.status = "Ready. Click Capture one sign."
        self.flip = False

    def capture(self, flip=False):
        with self.lock:
            self.flip = flip
            self.sequence = []
            self.tracked = 0
            self.ranked = []
            self.start_at = time.monotonic() + 1.5
            self.status = "Get ready — start signing when the countdown ends."

    def snapshot(self):
        with self.lock:
            return self.status, list(self.ranked), len(self.sequence)

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        with self.lock:
            now = time.monotonic()
            if self.start_at is not None:
                try:
                    if self.tracker is None:
                        self.tracker = make_tracker()
                    if now < self.start_at:
                        self.status = f"Start in {self.start_at - now:.1f}s"
                    elif now - self.last_frame_at >= 1 / 25:
                        self.last_frame_at = now
                        vector, image, valid = process_frame(self.tracker, image, self.flip)
                        self.sequence.append(vector)
                        self.tracked += int(valid)
                        self.status = f"Signing: {len(self.sequence)}/{FRAMES} frames"
                        if len(self.sequence) == FRAMES:
                            if self.tracked / FRAMES < 0.5:
                                self.status = "Tracking failed. Show upper body and hands; capture again."
                            else:
                                self.ranked = self.predictor.predict(self.sequence)
                                self.status = "Capture complete. Review prediction or capture another sign."
                            self.start_at = None
                except Exception as exc:
                    self.start_at = None
                    self.status = f"Camera processing failed: {exc}"
            cv2.putText(image, self.status[:75], (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 220, 255), 1)
        return av.VideoFrame.from_ndarray(image, format="bgr24")

    def on_ended(self):
        with self.lock:
            if self.tracker is not None:
                self.tracker.close()
                self.tracker = None
            self.start_at = None
