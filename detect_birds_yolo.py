#!/usr/bin/env python3
import os
import cv2
import time
import threading
from collections import deque
from datetime import datetime
import degirum as dg
import degirum_tools
from db import add_visit

# ==== Paths ====
CAPTURE_DIR = "images"
os.makedirs(CAPTURE_DIR, exist_ok=True)

# ==== RTSP from env ====
rtsp_user = os.environ.get("RTSP_USER")
rtsp_pass = os.environ.get("RTSP_PASS")
rtsp_host = os.environ.get("RTSP_HOST")
rtsp_port = os.environ.get("RTSP_PORT", "554")
rtsp_path = os.environ.get("RTSP_PATH")
if not all([rtsp_user, rtsp_pass, rtsp_host, rtsp_path]):
    raise EnvironmentError("Missing one or more RTSP environment variables.")

# Build RTSP URL and force TCP transport
VIDEO_SOURCE = (
    f"rtsp_transport=tcp;rtsp://{rtsp_user}:{rtsp_pass}@{rtsp_host}:{rtsp_port}/{rtsp_path}"
)
CAPTURE_OPTS = cv2.CAP_FFMPEG

# ==== Inference Config ====
CONFIDENCE_THRESHOLD = 0.6
COOLDOWN_SECONDS     = 10

# ==== Model Config ====
inference_host_address = "@local"
zoo_url                = "degirum/hailo"
token                  = ''
model_name             = "yolov8n_relu6_coco--640x640_quant_hailort_hailo8l_1"
output_class_set       = {"bird"}

# Load the AI model
model = dg.load_model(
    model_name=model_name,
    inference_host_address=inference_host_address,
    zoo_url=zoo_url,
    token=token,
    output_class_set=output_class_set
)

class RTSPBuffer:
    def __init__(self, src_url, buf_size=60, retry_delay=5):
        self.src_url     = src_url
        self.retry_delay = retry_delay
        self.buf         = deque(maxlen=buf_size)
        self.stopped     = False
        self.thread      = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        cap = None
        while not self.stopped:
            if cap is None or not cap.isOpened():
                if cap:
                    cap.release()
                print(f"[WARN] Can't open stream—retry in {self.retry_delay}s")
                time.sleep(self.retry_delay)
                cap = cv2.VideoCapture(self.src_url, CAPTURE_OPTS)
                continue

            ret, frame = cap.read()
            if ret:
                self.buf.append(frame)
            else:
                print("[WARN] Frame read failed—reopening capture")
                cap.release()
                cap = None
                time.sleep(0.1)

        if cap:
            cap.release()

    def read(self):
        return self.buf[-1] if self.buf else None

    def stop(self):
        self.stopped = True
        self.thread.join()


def monitor_rtsp():
    print("[INFO] Starting buffered RTSP monitor...")
    buffer = RTSPBuffer(VIDEO_SOURCE)
    last_ts = 0

    with degirum_tools.Display("Live Inference") as disp:
        while True:
            frame = buffer.read()
            if frame is None:
                continue

            # Run inference on the numpy frame directly
            result = degirum_tools.predict_frame(model, frame)
            disp.show(result)

            text = str(result).lower()
            if "object:" in text and "bird" in text:
                try:
                    conf = float(text.split("(")[-1].split(")")[0])
                except:
                    conf = 0.0
                if conf < CONFIDENCE_THRESHOLD:
                    continue

                now = time.time()
                if now - last_ts < COOLDOWN_SECONDS:
                    continue

                ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = f"bird_{ts}.jpg"
                path  = os.path.join(CAPTURE_DIR, fname)
                cv2.imwrite(path, frame)
                add_visit(
                    filename=fname,
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    species=None,
                    confidence=conf,
                    status="review",
                    classified=False
                )
                print(f"[CAPTURED] {fname}")
                last_ts = now

if __name__ == "__main__":
    try:
        monitor_rtsp()
    except KeyboardInterrupt:
        print("[INFO] Stopping monitor.")
