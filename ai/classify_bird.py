import onnxruntime as ort
from PIL import Image
import numpy as np
import datetime
import os
import sys
import shutil
import requests
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "app"))
from app import format_species_name
from db import add_visit

# Telegram debug
print("TELEGRAM_BOT_TOKEN:", os.getenv("TELEGRAM_BOT_TOKEN"))
print("TELEGRAM_CHAT_ID:", os.getenv("TELEGRAM_CHAT_ID"))

# Settings
CONFIDENCE_THRESHOLD = 0.65
REVIEW_THRESHOLD = 0.1
IMAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "images")
MODEL_PATH = "model/efficientnet_b7_backyard-birds.onnx"
LABELS_PATH = "model/class_labels.txt"

# Telegram settings
TELEGRAM_API_KEY = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Load model and labels
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name

with open(LABELS_PATH) as f:
    class_labels = [line.strip() for line in f]

# Ensure image folder exists
os.makedirs(IMAGE_DIR, exist_ok=True)

def send_telegram_message(message, image_path=None):
    if image_path:
        url = f"https://api.telegram.org/bot{TELEGRAM_API_KEY}/sendPhoto"
        payload = {'chat_id': CHAT_ID, 'caption': message}
        with open(image_path, 'rb') as photo:
            files = {'photo': photo}
            response = requests.post(url, data=payload, files=files)
    else:
        url = f"https://api.telegram.org/bot{TELEGRAM_API_KEY}/sendMessage"
        payload = {'chat_id': CHAT_ID, 'text': message}
        response = requests.post(url, data=payload)

    if response.status_code != 200:
        print(f"[ERROR] Telegram API returned status {response.status_code}: {response.text}")
    else:
        print("✅ Telegram notification sent.")

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB").resize((600, 600))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = (arr - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    arr = np.transpose(arr, (2, 0, 1))[np.newaxis, :]
    return arr.astype(np.float32)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def capture_and_classify(image_path, output_filename):
    # Extract timestamp from filename: "bird_2025-04-13_072653.jpg" or "motion_20250411_070035.jpg"
    basename = os.path.splitext(output_filename)[0]

    try:
        if basename.startswith("bird_"):
            timestamp = datetime.datetime.strptime(basename[5:], "%Y-%m-%d_%H%M%S")
        elif basename.startswith("motion_"):
            timestamp = datetime.datetime.strptime(basename[7:], "%Y%m%d_%H%M%S")
        else:
            raise ValueError("Unrecognized filename format")
    except Exception as e:
        print(f"[WARN] Failed to parse timestamp from filename: {e}")
        timestamp = datetime.datetime.now()

    timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")

    input_data = preprocess_image(image_path)
    output = session.run(None, {input_name: input_data})[0]
    probs = softmax(np.squeeze(output))
    prediction = np.argmax(probs)
    confidence = probs[prediction]
    species = class_labels[prediction]

    print(f"Predicted: {species} ({confidence:.2f})")
    normalized_species = species.strip().lower().replace(" ", "_")

    if normalized_species == "not_a_bird":
        print("[INFO] 'not_a_bird' detected — discarding image.")
        os.remove(image_path)
        sys.exit(2)

    if confidence < REVIEW_THRESHOLD:
        print("[INFO] Confidence too low, discarding image.")
        os.remove(image_path)
        sys.exit(2)

    # Determine classification status
    status = "accepted" if confidence >= CONFIDENCE_THRESHOLD else "review"

    # Notify if accepted
    if status == "accepted":
        pretty_species = format_species_name(species)
        send_telegram_message(f"A {pretty_species} has just visited your feeder!", image_path)

    # Move image to storage
    final_path = os.path.join(IMAGE_DIR, output_filename)
    shutil.move(image_path, final_path)

    # Create thumbnail
    thumb_dir = "thumbnails"
    os.makedirs(thumb_dir, exist_ok=True)
    thumb_path = os.path.join(thumb_dir, output_filename)
    with Image.open(final_path) as img:
        img.thumbnail((300, 169))
        img.save(thumb_path)

    # Log to database
    add_visit(
        filename=output_filename,
        timestamp=timestamp,
        species=species,
        confidence=round(float(confidence), 4),
        status=status
    )

    print(f"[DB] Stored {output_filename} as {status}")
    return status

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python classify_bird.py <image_path> <output_filename>")
        sys.exit(1)

    image_path = sys.argv[1]
    output_filename = sys.argv[2]

    result = capture_and_classify(image_path, output_filename)
    sys.exit(0 if result == "accepted" else 1)
