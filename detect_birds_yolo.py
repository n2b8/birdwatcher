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

    DETECTION_CMD_WITH_OUTPUT = DETECTION_CMD + ["--output", "captured_birds/latest.jpg"]
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
                source_path = "captured_birds/latest.jpg"
                raw_path = f"{CAPTURE_DIR}/raw_{timestamp}.jpg"
                final_filename = f"bird_{timestamp}.jpg"

                # Save detection to log file
                with open("detection_log.txt", "a") as log_file:
                    log_file.write(f"{timestamp} - {line}\n")

                # Try to capture latest.jpg (wait up to 0.5s)
                if os.path.exists(source_path):
                    os.rename(source_path, raw_path)
                    classify_bird(raw_path, final_filename)
                else:
                    print("[WARN] latest.jpg not found — waiting briefly...")
                    time.sleep(0.5)
                    if os.path.exists(source_path):
                        os.rename(source_path, raw_path)
                        classify_bird(raw_path, final_filename)
                    else:
                        print("[ERROR] Still no image found — skipping this detection.")

                time.sleep(3)  # Throttle detection frequency
    except KeyboardInterrupt:
        print("[INFO] Stopping YOLO monitor...")
        process.terminate()

if __name__ == "__main__":
    monitor_yolo()
