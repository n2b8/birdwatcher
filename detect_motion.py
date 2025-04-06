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
    filename = "/tmp/motion_frame.jpg"
    result = os.system(f"libcamera-still -n --width 640 --height 480 --quality 95 -o {filename}")
    if result == 0 and os.path.exists(filename):
        frame = cv2.imread(filename)
        return True, frame
    return False, None

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
        ret, frame = capture_frame()
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

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_path = f"visits/visit_{timestamp}.jpg"

                # Save full frame and classify
                cv2.imwrite(image_path, frame)
                print(f"[MOTION] Saved {image_path}, classifying...")

                result = subprocess.run(["python3", "classify_bird.py", image_path])
                if result.returncode != 0:
                    print("[INFO] No confident bird detected - image discarded.")
                    continue
                
                # Save debug frames if enabled
                if DEBUG and last_frame is not None:
                    cv2.imwrite("debug_last_frame.jpg", last_frame)
                    cv2.imwrite("debug_current_frame.jpg", small_frame)

        last_frame = small_frame
        time.sleep(0.5)  # Polling interval

except KeyboardInterrupt:
    print("[INFO] Exiting gracefully.")
