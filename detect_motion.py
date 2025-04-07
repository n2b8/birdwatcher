import cv2
import time
import os
import subprocess
from datetime import datetime

# Settings
COOLDOWN_SECONDS = 30
DEBUG = True  # Set to False to disable debug frame output

last_motion_time = 0
last_frame = None

def capture_frame():
    temp_path = "/tmp/motion_frame.jpg"
    result = os.system(f"libcamera-still -n --width 1920 --height 1080 --quality 95 -o {temp_path}")
    if result == 0 and os.path.exists(temp_path):
        frame = cv2.imread(temp_path)
        return True, frame, temp_path
    return False, None, None

def detect_motion(current, previous, threshold=30):
    if previous is None or current is None:
        return False

    diff = cv2.absdiff(previous, current)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    motion_score = cv2.countNonZero(thresh)

    print(f"Motion score: {motion_score}")
    return motion_score > 5000  # Adjust as needed

print("[INFO] Starting motion detection with libcamera-still...")

try:
    while True:
        ret, frame, temp_path = capture_frame()
        if not ret or frame is None:
            print("[ERROR] Frame not captured")
            time.sleep(1)
            continue
        else:
            print("[INFO] Frame captured")

        # Resize and compare frames
        small_frame = cv2.resize(frame, (400, 300))
        if detect_motion(small_frame, last_frame):
            now = time.time()
            if now - last_motion_time > COOLDOWN_SECONDS:
                last_motion_time = now

                print(f"[MOTION] Motion detected - classifying {temp_path}...")
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                unique_filename = f"motion_{timestamp}.jpg"
                result = subprocess.run(["python3", "classify_bird.py", temp_path, unique_filename])

                if result.returncode != 0:
                    print("[INFO] No confident bird detected - image discarded.")
                else:
                    print("[INFO] Bird image saved.")

                # Save debug frames if enabled
                if DEBUG and last_frame is not None:
                    cv2.imwrite("debug_last_frame.jpg", last_frame)
                    cv2.imwrite("debug_current_frame.jpg", small_frame)

        last_frame = small_frame
        time.sleep(0.5)

except KeyboardInterrupt:
    print("[INFO] Exiting gracefully.")
