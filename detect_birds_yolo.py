import subprocess
import time
import os
from datetime import datetime

# ==== YOLOv8 Detection Config ====
DETECTION_CMD = [
    "rpicam-hello",
    "-t", "0",
    "--post-process-file", "/usr/share/rpi-camera-assets/hailo_yolov8_inference.json",
    "-v", "2",
    "-n"
]

# ==== Paths ====
CAPTURE_DIR = "captured_birds"
os.makedirs(CAPTURE_DIR, exist_ok=True)

def capture_frame(raw_path):
    cmd = [
        "libcamera-still",
        "-o", raw_path,
        "-n",
        "--width", "1280",
        "--height", "720"
    ]
    subprocess.run(cmd)
    print(f"[INFO] Frame captured: {raw_path}")

def classify_bird(raw_path, filename, motion_score=1000):
    cmd = [
        "python3",
        "classify_bird.py",
        raw_path,
        filename,
        str(motion_score)
    ]
    subprocess.run(cmd)

def monitor_yolo():
    print("[INFO] Starting YOLOv8 monitor...")
    process = subprocess.Popen(
        DETECTION_CMD,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    try:
        for line in process.stdout:
            line = line.strip()
            if "bird" in line.lower():
                print(f"[DETECTED] {line}")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                raw_path = f"{CAPTURE_DIR}/raw_{timestamp}.jpg"
                final_filename = f"bird_{timestamp}.jpg"
                capture_frame(raw_path)
                classify_bird(raw_path, final_filename)
                time.sleep(3)  # throttle detections
    except KeyboardInterrupt:
        print("[INFO] Stopping...")
        process.terminate()

if __name__ == "__main__":
    monitor_yolo()
