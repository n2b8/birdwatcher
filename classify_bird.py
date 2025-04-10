import onnxruntime as ort
from PIL import Image
import numpy as np
import datetime
import os
import sys
import shutil
import requests

# Debugging telegram issues
print("TELEGRAM_BOT_TOKEN:", os.getenv("TELEGRAM_BOT_TOKEN"))
print("TELEGRAM_CHAT_ID:", os.getenv("TELEGRAM_CHAT_ID"))

# Settings
CONFIDENCE_THRESHOLD = 0.7
REVIEW_THRESHOLD = 0.1
TMP_DIR = "/tmp"
REVIEW_DIR = "review"
VISITS_DIR = "visits"
LOG_FILE = os.path.join(VISITS_DIR, "log.csv")
REVIEW_LOG = os.path.join(REVIEW_DIR, "review_log.csv")

# Telegram Settings
TELEGRAM_API_KEY = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Load model
session = ort.InferenceSession("model/efficientnet_b7_nabirds.onnx")
input_name = session.get_inputs()[0].name

# Load class labels
with open("model/class_labels_v2.txt") as f:
    class_labels = [line.strip() for line in f]

# Ensure necessary folders exist
os.makedirs(VISITS_DIR, exist_ok=True)
os.makedirs(REVIEW_DIR, exist_ok=True)

def send_telegram_message(message, image_path=None):
    if image_path:
        # Use the sendPhoto method with caption
        url = f"https://api.telegram.org/bot{TELEGRAM_API_KEY}/sendPhoto"
        payload = {'chat_id': CHAT_ID, 'caption': message}  # using 'caption' instead of 'text'
        with open(image_path, 'rb') as photo:
            files = {'photo': photo}
            response = requests.post(url, data=payload, files=files)
            print("Telegram sendPhoto response:", response.status_code, response.text)
    else:
        url = f"https://api.telegram.org/bot{TELEGRAM_API_KEY}/sendMessage"
        payload = {'chat_id': CHAT_ID, 'text': message}
        response = requests.post(url, data=payload)
        print("Telegram sendMessage response:", response.status_code, response.text)
        
    if response.status_code != 200:
        print(f"[ERROR] Telegram API returned status code {response.status_code}: {response.text}")

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB").resize((600, 600))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = (arr - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    arr = np.transpose(arr, (2, 0, 1))[np.newaxis, :]
    return arr.astype(np.float32)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def capture_and_classify(image_path, output_filename, motion_score=None):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    input_data = preprocess_image(image_path)
    output = session.run(None, {input_name: input_data})[0]
    probs = softmax(np.squeeze(output))
    prediction = np.argmax(probs)
    confidence = probs[prediction]
    species = class_labels[prediction]

    print(f"Predicted: {species} ({confidence:.2f})")

    # High confidence branch:
    if confidence >= CONFIDENCE_THRESHOLD:
        # If prediction is "not_a_bird", move to review folder instead of visits
        if species.lower() == "not_a_bird":
            final_path = os.path.join(REVIEW_DIR, output_filename)
            shutil.move(image_path, final_path)
            with open(REVIEW_LOG, "a", newline="") as f:
                if os.stat(REVIEW_LOG).st_size == 0:
                    f.write("filename,species,confidence,motion_score,timestamp\n")
                f.write(f"{output_filename},{species},{confidence:.2f},{motion_score},{timestamp}\n")
            return "review"
        else:
            # Accept the image and notify if it's a bird.
            message = f"A {species} has just visited your feeder!"
            send_telegram_message(message, image_path)
            final_path = os.path.join(VISITS_DIR, output_filename)
            shutil.move(image_path, final_path)
            with open(LOG_FILE, "a", newline="") as f:
                if os.stat(LOG_FILE).st_size == 0:
                    f.write("filename,species,confidence,motion_score,timestamp\n")
                f.write(f"{output_filename},{species},{confidence:.2f},{motion_score},{timestamp}\n")
            return "accepted"

    # Medium confidence: review
    elif confidence >= REVIEW_THRESHOLD:
        final_path = os.path.join(REVIEW_DIR, output_filename)
        shutil.move(image_path, final_path)
        with open(REVIEW_LOG, "a", newline="") as f:
            if os.stat(REVIEW_LOG).st_size == 0:
                f.write("filename,species,confidence,motion_score,timestamp\n")
            f.write(f"{output_filename},{species},{confidence:.2f},{motion_score},{timestamp}\n")
        return "review"

    # Low confidence: discard
    else:
        print("[INFO] Confidence too low, discarding image.")
        os.remove(image_path)
        return "discarded"

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python classify_bird.py <image_path> <output_filename> <motion_score>")
        sys.exit(1)

    image_path = sys.argv[1]
    output_filename = sys.argv[2]
    motion_score = sys.argv[3]

    result = capture_and_classify(image_path, output_filename, motion_score)
    sys.exit(0 if result == "accepted" else 1)
