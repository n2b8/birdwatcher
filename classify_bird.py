import onnxruntime as ort
from PIL import Image
import numpy as np
import datetime
import os
import sys
from send_telegram import send_telegram_message

# Settings
CONFIDENCE_THRESHOLD = 2.0  # Minimum probability to consider a valid bird
VISITS_DIR = "visits"
LOG_FILE = os.path.join(VISITS_DIR, "log.csv")

# Load ONNX model
session = ort.InferenceSession("model/efficientnet_b0_nabirds.onnx")
input_name = session.get_inputs()[0].name

# Load class labels
with open("model/class_labels.txt") as f:
    class_labels = [line.strip() for line in f]

# Ensure visits folder exists
os.makedirs(VISITS_DIR, exist_ok=True)

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = (arr - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    arr = np.transpose(arr, (2, 0, 1))[np.newaxis, :]
    return arr.astype(np.float32)

def capture_and_classify(image_path=None):
    if image_path is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(VISITS_DIR, f"visit_{timestamp}.jpg")
        os.system(f"libcamera-still -n -o {image_path} --width 640 --height 480")
    else:
        timestamp = image_path.split("_")[-1].split(".")[0]

    # Preprocess and classify
    input_data = preprocess_image(image_path)
    output = session.run(None, {input_name: input_data})[0]
    probs = np.squeeze(output)
    prediction = np.argmax(probs)
    confidence = probs[prediction]

    if confidence >= CONFIDENCE_THRESHOLD:
        species = class_labels[prediction]
        print(f"[{timestamp}] ✅ Detected: {species} ({confidence:.2f})")

        # Write to log
        with open(LOG_FILE, "a", newline="") as log:
            if os.stat(LOG_FILE).st_size == 0:
                log.write("filename,species,timestamp\n")
            log.write(f"{os.path.basename(image_path)},{species},{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Send Telegram notification
        send_telegram_message(species, image_path)

        return True
    else:
        print(f"[{timestamp}] ❌ Low confidence ({confidence:.2f}), skipping image.")
        os.remove(image_path)
        return False

if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    success = capture_and_classify(image_path)
    sys.exit(0 if success else 1)
