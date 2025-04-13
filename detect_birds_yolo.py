import subprocess
import time
import os
from datetime import datetime
from db import add_visit

# ==== Paths ====
CAPTURE_DIR = "images"
os.makedirs(CAPTURE_DIR, exist_ok=True)

# ==== Detection Config ====
DETECTION_CMD = [
    "rpicam-hello",
    "-t", "0",
    "--post-process-file", "/usr/share/rpi-camera-assets/hailo_yolov8_inference.json",
    "-v", "2",
    "-n"
]

CONFIDENCE_THRESHOLD = 0.6
COOLDOWN_SECONDS = 10

def capture_frame(path):
    result = subprocess.run([
        "libcamera-still",
        "-o", path,
        "-n",
        "--width", "1280",
        "--height", "720"
    ])
    return result.returncode == 0 and os.path.exists(path)

def monitor_yolo():
    print("[INFO] Starting YOLOv8 bird monitor loop...")

    while True:
        print("[INFO] Launching rpicam-hello...")
        process = subprocess.Popen(
            DETECTION_CMD,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        try:
            for line in process.stdout:
                line = line.strip()

                if "object:" in line.lower():
                    print("[DETECTION]", line)

                # Extract bird detection
                if "object:" in line.lower() and "bird" in line.lower():
                    try:
                        confidence = float(line.split("(")[-1].split(")")[0])
                        if confidence < CONFIDENCE_THRESHOLD:
                            continue
                    except Exception:
                        continue

                    print("[DETECTED BIRD]", line)
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    filename = f"bird_{timestamp.replace(':','').replace(' ', '_')}.jpg"
                    filepath = os.path.join(CAPTURE_DIR, filename)

                    # Kill detection process
                    process.terminate()
                    process.wait()
                    print("[INFO] YOLO detection stopped for still capture.")

                    if capture_frame(filepath):
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
                        print("[ERROR] Failed to capture image.")

                    print(f"[WAIT] Cooling down for {COOLDOWN_SECONDS} seconds...\n")
                    time.sleep(COOLDOWN_SECONDS)
                    break  # restart loop after cooldown

        except KeyboardInterrupt:
            print("[INFO] Shutting down bird detector.")
            process.terminate()
            break

if __name__ == "__main__":
    monitor_yolo()
