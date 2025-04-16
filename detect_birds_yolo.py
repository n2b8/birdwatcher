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

# ==== Environment Variables for RTSP ====
rtsp_user = os.environ.get("RTSP_USER")
rtsp_pass = os.environ.get("RTSP_PASS")
rtsp_host = os.environ.get("RTSP_HOST")
rtsp_port = os.environ.get("RTSP_PORT", "554")
rtsp_path = os.environ.get("RTSP_PATH")
if not all([rtsp_user, rtsp_pass, rtsp_host, rtsp_path]):
    raise EnvironmentError("Missing RTSP environment variables.")

# Force TCP transport to reduce packet loss
VIDEO_SOURCE = f"rtsp_transport=tcp;rtsp://{rtsp_user}:{rtsp_pass}@{rtsp_host}:{rtsp_port}/{rtsp_path}"

# ==== Inference Config ====
CONFIDENCE_THRESHOLD = 0.6
COOLDOWN_SECONDS = 10

# ==== Model Configuration ====
inference_host_address = "@local"
zoo_url = "degirum/hailo"
token = ''  # local inference
model_name = "yolov8n_relu6_coco--640x640_quant_hailort_hailo8_1"
output_class_set = {"bird"}

# Load the AI model
model = dg.load_model(
    model_name=model_name,
    inference_host_address=inference_host_address,
    zoo_url=zoo_url,
    token=token,
    output_class_set=output_class_set
)

class RTSPBuffer:
    """Background thread to fill buffer with the latest frames."""
    def __init__(self, src_url, buf_size=60):
        self.src_url = src_url
        self.buf = deque(maxlen=buf_size)
        self.stopped = False
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        cap = cv2.VideoCapture(self.src_url)
        if not cap.isOpened():
            raise RuntimeError("Cannot open RTSP stream")
        while not self.stopped:
            ret, frame = cap.read()
            if ret:
                self.buf.append(frame)
            else:
                time.sleep(0.01)
        cap.release()

    def read(self):
        return self.buf[-1] if self.buf else None

    def stop(self):
        self.stopped = True
        self.thread.join()


def monitor_yolo_rtsp():
    print("[INFO] Starting buffered RTSP bird monitor...")
    last_detection = 0

    # Initialize buffer (~2s at 30fps)
    buffer = RTSPBuffer(VIDEO_SOURCE, buf_size=60)
    try:
        with degirum_tools.Display("AI Camera Live Inference") as display:
            while True:
                frame = buffer.read()
                if frame is None:
                    continue

                # Run inference on the numpy frame directly
                inference_result = degirum_tools.predict_frame(model, frame)
                display.show(inference_result)

                text = str(inference_result).lower()
                if "object:" in text and "bird" in text:
                    try:
                        confidence = float(text.split("(")[-1].split(")")[0])
                    except:
                        confidence = 0.0
                    if confidence < CONFIDENCE_THRESHOLD:
                        continue

                    now = time.time()
                    if now - last_detection < COOLDOWN_SECONDS:
                        continue

                    print("[DETECTED BIRD]", inference_result)
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    fname = f"bird_{ts.replace(':','').replace(' ','_')}.jpg"
                    path = os.path.join(CAPTURE_DIR, fname)

                    # Save the last buffered frame
                    cv2.imwrite(path, frame)
                    add_visit(filename=fname, timestamp=ts, species=None,
                              confidence=confidence, status="review", classified=False)
                    print(f"[CAPTURED] {fname}")

                    last_detection = now
                    print(f"[WAIT] Cooling down for {COOLDOWN_SECONDS}s\n")
    except KeyboardInterrupt:
        pass
    finally:
        buffer.stop()


if __name__ == "__main__":
    monitor_yolo_rtsp()
