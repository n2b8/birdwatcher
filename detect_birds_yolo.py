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
    result = subprocess.run(cmd)
    if result.returncode != 0 or not os.path.exists(raw_path):
        print(f"[ERROR] Failed to capture image at {raw_path}")
        return None
    print(f"[INFO] Frame captured: {raw_path}")
    return raw_path

def classify_bird(raw_path, filename, motion_score=1000):
    if not raw_path:
        print("[WARN] No image to classify.")
        return
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

    DETECTION_CMD_WITH_OUTPUT = DETECTION_CMD  # <-- Removed --output
    print("[DEBUG] Full detection command:", " ".join(DETECTION_CMD_WITH_OUTPUT))

    process = subprocess.Popen(
        DETECTION_CMD_WITH_OUTPUT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    if process.stdout is None:
        print("[ERROR] Failed to open rpicam-hello process.")
        return

    try:
        for line in process.stdout:
            line = line.strip()

            # Show only detection lines
            if "object:" in line.lower():
                print("[DETECTION]", line)

            # Trigger on bird only
            if "object:" in line.lower() and "bird" in line.lower():
                print("[DETECTED BIRD]", line)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                raw_path = f"{CAPTURE_DIR}/raw_{timestamp}.jpg"
                final_filename = f"bird_{timestamp}.jpg"

                # Save detection to log file
                with open("detection_log.txt", "a") as log_file:
                    log_file.write(f"{timestamp} - {line}\n")

                # Capture a fresh image
                raw_path = capture_frame(raw_path)
                classify_bird(raw_path, final_filename)

                time.sleep(3)  # Throttle detection frequency
    except KeyboardInterrupt:
        print("[INFO] Stopping YOLO monitor...")
        process.terminate()

if __name__ == "__main__":
    monitor_yolo()
