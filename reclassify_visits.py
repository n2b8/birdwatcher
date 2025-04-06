import onnxruntime as ort
from PIL import Image
import numpy as np
import os
import csv
from datetime import datetime

# Settings
VISITS_DIR = "visits"
LOG_FILE = os.path.join(VISITS_DIR, "log.csv")
MODEL_PATH = "model/efficientnet_b0_nabirds.onnx"
LABELS_PATH = "model/class_labels.txt"
CONFIDENCE_THRESHOLD = 1.0

# Load model
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name

# Load labels
with open(LABELS_PATH) as f:
    class_labels = [line.strip() for line in f]

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = (arr - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    arr = np.transpose(arr, (2, 0, 1))[np.newaxis, :]
    return arr.astype(np.float32)

# Reclassify images and rewrite log
new_entries = []

for filename in sorted(os.listdir(VISITS_DIR)):
    if not filename.endswith(".jpg"):
        continue
    image_path = os.path.join(VISITS_DIR, filename)
    try:
        input_data = preprocess_image(image_path)
        output = session.run(None, {input_name: input_data})[0]
        probs = np.squeeze(output)
        prediction = np.argmax(probs)
        confidence = probs[prediction]

        if confidence >= CONFIDENCE_THRESHOLD:
            species = class_labels[prediction]
            timestamp_str = filename.replace("visit_", "").replace(".jpg", "")
            timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            new_entries.append([filename, species, timestamp.strftime("%Y-%m-%d %H:%M:%S")])
            print(f"[✅] {filename}: {species} ({confidence:.2f})")
        else:
            os.remove(image_path)
            print(f"[❌] {filename}: Low confidence ({confidence:.2f}) — deleted.")

    except Exception as e:
        print(f"[ERROR] Failed to process {filename}: {e}")

# Rewrite log.csv
with open(LOG_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "species", "timestamp"])
    writer.writerows(new_entries)

print("✅ Reclassification complete.")
