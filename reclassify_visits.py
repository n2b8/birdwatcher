import os
import csv
from datetime import datetime
from PIL import Image
import numpy as np
import onnxruntime as ort
import torchvision.transforms as transforms

VISITS_DIR = "visits"
MODEL_PATH = "model/efficientnet_b0_nabirds.onnx"
LABELS_PATH = "model/class_labels.txt"
LOG_CSV = os.path.join(VISITS_DIR, "log.csv")

# Load labels
with open(LABELS_PATH, "r") as f:
  labels = [line.strip() for line in f]

# ONNX model session
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name

# Image transform
transform = transforms.Compose([
  transforms.Resize((224, 224)),
  transforms.ToTensor(),
  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Reclassify all images
entries = []
for fname in sorted(os.listdir(VISITS_DIR)):
  if not fname.endswith(".jpg"):
      continue

  image_path = os.path.join(VISITS_DIR, fname)
  image = Image.open(image_path).convert("RGB")
  tensor = transform(image).unsqueeze(0).numpy().astype(np.float32)

  output = session.run(None, {input_name: tensor})[0]
  pred = np.argmax(output)
  label = labels[pred]

  # Get timestamp from filename
  ts_str = fname.replace("visit_", "").replace(".jpg", "")
  try:
      ts = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
  except ValueError:
      ts = datetime.now()

  entries.append([fname, label, ts.strftime("%Y-%m-%d %H:%M:%S")])
  print(f"{fname} ➜ {label}")

# Overwrite log.csv
with open(LOG_CSV, "w", newline="") as f:
  writer = csv.writer(f)
  writer.writerow(["filename", "species", "timestamp"])
  writer.writerows(entries)

print("✅ Reclassification complete.")
