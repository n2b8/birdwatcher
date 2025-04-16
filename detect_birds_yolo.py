import os
import degirum as dg
import degirum_tools
import cv2
import time
from datetime import datetime
from db import add_visit

# ==== Paths ====
CAPTURE_DIR = "images"
os.makedirs(CAPTURE_DIR, exist_ok=True)

# ==== Environment Variables for RTSP ====
# These variables should be set in your systemd service file:
# RTSP_USER, RTSP_PASS, RTSP_PORT, RTSP_HOST, RTSP_PATH
rtsp_user = os.environ.get("RTSP_USER")
rtsp_pass = os.environ.get("RTSP_PASS")
rtsp_host = os.environ.get("RTSP_HOST")
rtsp_port = os.environ.get("RTSP_PORT", "554")  # default to 554 if not provided
rtsp_path = os.environ.get("RTSP_PATH")

if not all([rtsp_user, rtsp_pass, rtsp_host, rtsp_path]):
    raise EnvironmentError("Missing one or more RTSP environment variables.")

VIDEO_SOURCE = f"rtsp://{rtsp_user}:{rtsp_pass}@{rtsp_host}:{rtsp_port}/{rtsp_path}"

# ==== Inference Config ====
CONFIDENCE_THRESHOLD = 0.6
COOLDOWN_SECONDS = 10

# ==== Model Configuration ====
inference_host_address = "@local"
zoo_url = "degirum/hailo"
token = ''  # Leave empty for local inference
model_name = "yolov8n_relu6_coco--640x640_quant_hailort_hailo8_1"
output_class_set = {"bird"}  # Only process detections for birds

# Load the AI model via degirum
model = dg.load_model(
    model_name=model_name,
    inference_host_address=inference_host_address,
    zoo_url=zoo_url,
    token=token,
    output_class_set=output_class_set
)

def capture_rtsp_frame(video_source, output_path):
    """
    Opens the RTSP stream, grabs one frame, saves it to the given path,
    and then releases the capture.
    """
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Cannot open RTSP stream for capturing frame.")
        return False
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(output_path, frame)
        cap.release()
        return True
    cap.release()
    return False

def monitor_yolo_rtsp():
    print("[INFO] Starting YOLOv8 bird monitor loop using RTSP...")
    last_detection_time = 0

    # Use degirum_tools.Display to show inference results in a window
    with degirum_tools.Display("AI Camera Live Inference") as output_display:
        # Iterate over inference results from the RTSP stream
        for inference_result in degirum_tools.predict_stream(model, VIDEO_SOURCE):
            output_display.show(inference_result)

            # In this example, we assume inference_result can be converted to a string
            # that includes "object:" along with the object name and confidence.
            text = str(inference_result).lower()

            if "object:" in text and "bird" in text:
                try:
                    # Extract the confidence value (assuming it's in parentheses at the end)
                    confidence = float(text.split("(")[-1].split(")")[0])
                except Exception:
                    confidence = 0.0

                # Only continue if the confidence meets the threshold
                if confidence < CONFIDENCE_THRESHOLD:
                    continue

                # Enforce cooldown to prevent multiple captures in a short time
                current_time = time.time()
                if (current_time - last_detection_time) < COOLDOWN_SECONDS:
                    continue

                print("[DETECTED BIRD]", inference_result)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                filename = f"bird_{timestamp.replace(':', '').replace(' ', '_')}.jpg"
                filepath = os.path.join(CAPTURE_DIR, filename)

                # Capture a single frame from the RTSP stream using OpenCV
                if capture_rtsp_frame(VIDEO_SOURCE, filepath):
                    print(f"[CAPTURED] {filename}")
                    add_visit(
                        filename=filename,
                        timestamp=timestamp,
                        species=None,
                        confidence=confidence,
                        status="review",
                        classified=False
                    )
                else:
                    print("[ERROR] Failed to capture image from RTSP stream.")

                print(f"[WAIT] Cooling down for {COOLDOWN_SECONDS} seconds...\n")
                last_detection_time = time.time()

if __name__ == "__main__":
    try:
        monitor_yolo_rtsp()
    except KeyboardInterrupt:
        print("[INFO] Shutting down bird detector.")
