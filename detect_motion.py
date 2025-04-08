import cv2
import time
import os
import subprocess
from datetime import datetime
import numpy as np

# Settings
COOLDOWN_SECONDS = 30
BRIGHTNESS_THRESHOLD = 100  # Set the brightness threshold
MOTION_THRESHOLD = 10000    # Set the motion threshold
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

def calculate_brightness(image):
    # Convert to grayscale and calculate the mean brightness
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

def detect_motion(current, previous, threshold=30, motion_threshold=MOTION_THRESHOLD):
    if previous is None or current is None:
        return 0  # Return the motion score as 0 when there is no previous frame

    diff = cv2.absdiff(previous, current)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    motion_score = cv2.countNonZero(thresh)

    print(f"Motion score: {motion_score}")

    # Compare against the motion threshold
    if motion_score > motion_threshold:
        return motion_score  # Return motion score if it exceeds the threshold
    else:
        return 0  # Return 0 if motion score is below threshold

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

        # Calculate the brightness of the frame
        brightness = calculate_brightness(frame)
        print(f"Image brightness: {brightness}")

        # If the brightness is below the threshold, discard the image
        if brightness < BRIGHTNESS_THRESHOLD:
            print("[INFO] Image brightness is too low, discarding image.")
            continue  # Skip this image and move to the next one

        # Resize and compare frames
        small_frame = cv2.resize(frame, (640, 480))
        motion_score = detect_motion(small_frame, last_frame)  # Get the motion score
        if motion_score > MOTION_THRESHOLD:  # If the motion score is large enough, proceed to classify
            now = time.time()
            if now - last_motion_time > COOLDOWN_SECONDS:
                last_motion_time = now

                print(f"[MOTION] Motion detected - classifying {temp_path}...")
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                unique_filename = f"motion_{timestamp}.jpg"

                # 👇 Pass environment variables to subprocess
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

                # Save debug frames if enabled
                if DEBUG and last_frame is not None:
                    cv2.imwrite("debug_last_frame.jpg", last_frame)
                    cv2.imwrite("debug_current_frame.jpg", small_frame)

        last_frame = small_frame
        time.sleep(0.5)

except KeyboardInterrupt:
    print("[INFO] Exiting gracefully.")
