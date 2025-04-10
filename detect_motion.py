import cv2
import time
import os
import subprocess
from datetime import datetime
import numpy as np

# Settings
COOLDOWN_SECONDS = 30
BRIGHTNESS_THRESHOLD = 85
MOTION_THRESHOLD = 10000
DEBUG = True
MOTION_STABILIZATION_THRESHOLD = 0.9

last_motion_time = 0
last_frame = None

def stabilize_frame(prev_gray, curr_gray, cc_threshold=MOTION_STABILIZATION_THRESHOLD):
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    try:
        cc, warp_matrix = cv2.findTransformECC(prev_gray, curr_gray, warp_matrix, cv2.MOTION_EUCLIDEAN)
        if cc < cc_threshold:
            print(f"[WARN] Stabilization confidence too low: {cc:.2f}")
            return None
        stabilized = cv2.warpAffine(curr_gray, warp_matrix, (curr_gray.shape[1], curr_gray.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        return stabilized
    except cv2.error as e:
        print(f"[WARN] Stabilization failed: {e}")
        return None

def capture_frame():
    temp_path = "/tmp/motion_frame.jpg"
    result = os.system(f"libcamera-still -n --width 1920 --height 1080 --quality 95 -o {temp_path}")
    if result == 0 and os.path.exists(temp_path):
        frame = cv2.imread(temp_path)
        return True, frame, temp_path
    return False, None, None

def calculate_brightness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

def detect_motion(current, previous, threshold=30, motion_threshold=MOTION_THRESHOLD):
    if previous is None or current is None:
        return 0

    diff = cv2.absdiff(previous, current)
    gray = diff if len(diff.shape) == 2 else cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    motion_score = cv2.countNonZero(thresh)

    print(f"Motion score: {motion_score}")
    return motion_score if motion_score > motion_threshold else 0

print("[INFO] Starting motion detection with libcamera-still...")

try:
    while True:
        ret, frame, temp_path = capture_frame()
        if not ret or frame is None:
            print("[ERROR] Frame not captured")
            time.sleep(1)
            continue

        print("[INFO] Frame captured")
        brightness = calculate_brightness(frame)
        print(f"Image brightness: {brightness}")

        if brightness < BRIGHTNESS_THRESHOLD:
            print("[INFO] Image brightness is too low, discarding image.")
            continue

        small_frame = cv2.resize(frame, (640, 480))
        gray_current = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        gray_prev = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY) if last_frame is not None else None

        if gray_prev is not None:
            stabilized_current = stabilize_frame(gray_prev, gray_current)
            motion_score = detect_motion(stabilized_current, gray_prev) if stabilized_current is not None else 0
        else:
            motion_score = 0

        if motion_score > MOTION_THRESHOLD:
            now = time.time()
            if now - last_motion_time > COOLDOWN_SECONDS:
                last_motion_time = now
                print(f"[MOTION] Motion detected - classifying {temp_path}...")
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                unique_filename = f"motion_{timestamp}.jpg"

                env = os.environ.copy()
                env["TELEGRAM_BOT_TOKEN"] = os.getenv("TELEGRAM_BOT_TOKEN", "")
                env["TELEGRAM_CHAT_ID"] = os.getenv("TELEGRAM_CHAT_ID", "")

                result = subprocess.run(
                    ["python3", "classify_bird.py", temp_path, unique_filename, str(motion_score)],
                    env=env
                )

                if result.returncode != 0:
                    print("[INFO] No confident bird detected - image discarded.")
                else:
                    print("[INFO] Bird image saved.")

                if DEBUG and last_frame is not None:
                    cv2.imwrite("debug_last_frame.jpg", last_frame)
                    cv2.imwrite("debug_current_frame.jpg", small_frame)

        last_frame = small_frame
        time.sleep(0.5)

except KeyboardInterrupt:
    print("[INFO] Exiting gracefully.")
