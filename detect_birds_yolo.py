import cv2
import time
import os
from hailo_platform import HEF
from hailo_model_zoo.core.infer import infer_model
from hailo_model_zoo.core.postprocessing.detection_postprocessing import decode_detections
from hailo_model_zoo.utils.yaml_utilities import YamlLoader

# ==== Paths ====
HEF_PATH = "model/hailo/yolov8s/yolov8s.hef"
YAML_PATH = "model/hailo/yolov8s/yolov8s.yaml"
OUTPUT_DIR = "captured_birds"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==== Inference Settings ====
BIRD_CLASS_ID = 14  # COCO index for 'bird'
CONFIDENCE_THRESHOLD = 0.5

# ==== Load YAML config ====
yaml_loader = YamlLoader(YAML_PATH)
network_info, infer_config, postproc_config = yaml_loader.load_yaml()

# ==== Init Model ====
print("[INFO] Loading YOLOv8s model...")
model = infer_model(
    hef_path=HEF_PATH,
    network_info=network_info,
    input_shape=network_info['input_shape']
)
print("[INFO] Model loaded.")

# ==== Open Webcam ====
cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
if not cap.isOpened():
    raise RuntimeError("Unable to open camera")

print("[INFO] Starting video capture... Press Ctrl+C to stop.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame")
            continue

        # Resize frame to model input
        resized = cv2.resize(frame, (640, 640))  # yolov8s is 640x640
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Run inference
        results = model.infer(rgb)

        # Decode results
        decoded = decode_detections(results, postproc_config, CONFIDENCE_THRESHOLD)

        # Check for birds
        bird_detected = any(obj['class_id'] == BIRD_CLASS_ID for obj in decoded)
        if bird_detected:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{OUTPUT_DIR}/bird_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"[INFO] Bird detected — saved to {filename}")
            time.sleep(2)  # Avoid spamming multiple detections

except KeyboardInterrupt:
    print("\n[INFO] Exiting gracefully...")

finally:
    cap.release()
    cv2.destroyAllWindows()
