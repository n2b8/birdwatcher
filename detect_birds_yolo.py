import os
import time
import cv2
import degirum as dg
import degirum_tools
from datetime import datetime
from db import add_visit

# ==== RTSP from env ====
rtsp_user = os.environ["RTSP_USER"]
rtsp_pass = os.environ["RTSP_PASS"]
rtsp_host = os.environ["RTSP_HOST"]
rtsp_port = os.environ.get("RTSP_PORT", "554")
rtsp_path = os.environ["RTSP_PATH"]

# —— CONFIG —— #
VIDEO_SOURCE = f"rtsp://{rtsp_user}:{rtsp_pass}@{rtsp_host}:{rtsp_port}/{rtsp_path}"
CONFIDENCE_THRESHOLD = 0.6
COOLDOWN_SECONDS     = 10
CAPTURE_DIR          = "images"
MODEL_NAME           = "yolov8n_relu6_coco--640x640_quant_hailort_hailo8_1"
INFERENCE_HOST       = "@local"
ZOO_URL              = "degirum/models_hailort"
TOKEN                = ""   # empty for local
OUTPUT_CLASS_SET     = {"bird"}
# —————— #

os.makedirs(CAPTURE_DIR, exist_ok=True)

# Load the bird‑only YOLOv8 model
model = dg.load_model(
    model_name=MODEL_NAME,
    inference_host_address=INFERENCE_HOST,
    zoo_url=ZOO_URL,
    token=TOKEN,
    output_class_set=OUTPUT_CLASS_SET
)

def capture_frame(rtsp_url, path):
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        return False
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return False
    cv2.imwrite(path, frame)
    return True

def monitor():
    print("[INFO] Starting bird monitor…")
    last_time = 0.0

    for res in degirum_tools.predict_stream(model, VIDEO_SOURCE):
        # res.results is a flat list of dicts
        for det in res.results:
            if det["label"] != "bird":
                continue
            if det["score"] < CONFIDENCE_THRESHOLD:
                continue

            now = time.time()
            if now - last_time < COOLDOWN_SECONDS:
                continue

            # passed all filters — capture & log
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            filename  = f"bird_{timestamp.replace(':','').replace(' ','_')}.jpg"
            filepath  = os.path.join(CAPTURE_DIR, filename)

            if capture_frame(VIDEO_SOURCE, filepath):
                print(f"[CAPTURED] {filename} @ {det['score']*100:.1f}%")
                add_visit(
                    filename=filename,
                    timestamp=timestamp,
                    species=None,             # you can fill this in later
                    status="review",
                    classified=False
                )
                last_time = now
            else:
                print("[ERROR] failed to grab still image")

if __name__ == "__main__":
    try:
        monitor()
    except KeyboardInterrupt:
        print("[INFO] Shutting down.")
