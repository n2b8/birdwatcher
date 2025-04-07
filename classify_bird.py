import onnxruntime as ort
from PIL import Image
import numpy as np
import datetime
import os
import sys
import shutil

# Settings
CONFIDENCE_THRESHOLD = 2.0
REVIEW_THRESHOLD = 0.6
TMP_DIR = "/tmp"
REVIEW_DIR = "review"
VISITS_DIR = "visits"
LOG_FILE = os.path.join(VISITS_DIR, "log.csv")
REVIEW_LOG = os.path.join(REVIEW_DIR, "review_log.csv")

# Load model
session = ort.InferenceSession("model/efficientnet_b0_nabirds.onnx")
input_name = session.get_inputs()[0].name

# Load class labels
with open("model/class_labels.txt") as f:
    class_labels = [line.strip() for line in f]

# Ensure necessary folders exist
os.makedirs(VISITS_DIR, exist_ok=True)
os.makedirs(REVIEW_DIR, exist_ok=True)

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = (arr - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    arr = np.transpose(arr, (2, 0, 1))[np.newaxis, :]
    return arr.astype(np.float32)

def capture_and_classify(image_path, output_filename):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    input_data = preprocess_image(image_path)
    output = session.run(None, {input_name: input_data})[0]
    probs = np.squeeze(output)
    prediction = np.argmax(probs)
    confidence = probs[prediction]
    species = class_labels[prediction]

    print(f"Predicted: {species} ({confidence:.2f})")

    if confidence >= CONFIDENCE_THRESHOLD:
        final_path = os.path.join(VISITS_DIR, output_filename)
        shutil.move(image_path, final_path)
        with open(LOG_FILE, "a", newline="") as f:
            if os.stat(LOG_FILE).st_size == 0:
                f.write("filename,species,timestamp\n")
            f.write(f"{output_filename},{species},{timestamp}\n")
        return "accepted"

    else:
        final_path = os.path.join(REVIEW_DIR, output_filename)
        shutil.move(image_path, final_path)
        with open(REVIEW_LOG, "a", newline="") as f:
            if os.stat(REVIEW_LOG).st_size == 0:
                f.write("filename,species,confidence,timestamp\n")
            f.write(f"{output_filename},{species},{confidence:.2f},{timestamp}\n")
        return "review"

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python classify_bird.py <image_path> <output_filename>")
        sys.exit(1)

    image_path = sys.argv[1]
    output_filename = sys.argv[2]

    result = capture_and_classify(image_path, output_filename)
    sys.exit(0 if result == "accepted" else 1)
