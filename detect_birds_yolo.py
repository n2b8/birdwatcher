#!/usr/bin/env python3
import os
import cv2
import time
from datetime import datetime
import degirum as dg
import degirum_tools
from db import add_visit

# ==== Paths ====
CAPTURE_DIR = "images"
os.makedirs(CAPTURE_DIR, exist_ok=True)

# ==== RTSP from env ====
rtsp_user = os.environ["RTSP_USER"]
rtsp_pass = os.environ["RTSP_PASS"]
rtsp_host = os.environ["RTSP_HOST"]
rtsp_port = os.environ.get("RTSP_PORT", "554")
rtsp_path = os.environ["RTSP_PATH"]
VIDEO_SOURCE = f"rtsp://{rtsp_user}:{rtsp_pass}@{rtsp_host}:{rtsp_port}/{rtsp_path}"

# ==== Inference Config ====
CONFIDENCE_THRESHOLD = 0.6
COOLDOWN_SECONDS = 10

# ==== Model Config ====
model = dg.load_model(
    model_name="yolov8n_relu6_coco--640x640_quant_hailort_hailo8_1",
    inference_host_address="@local",
    zoo_url="degirum/hailo",
    token="",
    device_type=["HAILORT/HAILO8"],
    output_class_set={"bird"},
)

def monitor_rtsp():
    print("[INFO] Starting RTSP stream inference (no window)...")
    last_ts = 0.0

    for result in degirum_tools.predict_stream(model, VIDEO_SOURCE):
        # result.frame is the cv2 image, result.objects is a list of detections
        frame = result.frame
        detections = result.objects  # <- this is a Python list

        if frame is None:
            # fallback grab
            cap = cv2.VideoCapture(VIDEO_SOURCE)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                continue

        for det in detections:
            # det.label, det.score, det.bbox, etc.
            if det.label != "bird" or det.score is None:
                continue
            if det.score < CONFIDENCE_THRESHOLD:
                continue

            now = time.time()
            if now - last_ts < COOLDOWN_SECONDS:
                break

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"bird_{ts}.jpg"
            path = os.path.join(CAPTURE_DIR, fname)

            cv2.imwrite(path, frame)
            add_visit(
                filename=fname,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                species=None,
                confidence=det.score,
                status="review",
                classified=False
            )
            print(f"[CAPTURED] {fname} (score={det.score:.2f})")

            last_ts = now
            break  # one capture per frame


if __name__ == "__main__":
    try:
        monitor_rtsp()
    except KeyboardInterrupt:
        print("[INFO] Stopping monitor.")
