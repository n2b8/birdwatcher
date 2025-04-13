# Birdwatcher 🐦📸

A Raspberry Pi-powered bird identification system using real-time YOLO object detection and EfficientNet-based species classification. Detects birds with hardware acceleration, logs visits with metadata to SQLite, and provides a responsive web interface for viewing and stats.

---

## Features

- 🦾 **YOLOv8 (Hailo)** real-time bird detection using `rpicam-hello`
- 📷 High-quality still capture on detection
- 🧠 Custom EfficientNet-B7 classification (ONNX model, trained on a subset of the [NABirds dataset](https://dl.allaboutbirds.org/nabirds))
- 🗂️ Images + metadata stored in SQLite (`visits` table)
- 🔍 Review interface for uncertain predictions
- 📊 Species frequency charts + time-based heatmap
- 📡 Telegram notifications on high-confidence visits
- 🛠️ systemd services for background classification & detection

---

## Project Structure

```
.
├── app.py                     # Flask web dashboard
├── classify_bird.py           # Classify image using ONNX model
├── classify_queue.py          # Background classification queue processor
├── detect_birds_yolo.py       # YOLOv8 Hailo detection loop (rpicam-hello)
├── db.py                      # SQLite connection + visit helpers
├── send_telegram.py           # Telegram notification module
├── test_telegram.py           # Test bot integration
├── model/
│   ├── class_labels_v2.txt
│   └── efficientnet_b7_nabirds.onnx
├── images/                    # Accepted/candidate images
├── thumbnails/                # Web-optimized thumbnails
├── static/                    # Generated charts
├── templates/                 # Flask HTML views
│   ├── index.html
│   ├── review.html
│   └── stats.html
└── .gitignore
```

---

## Setup

```bash
# Clone the repo
git clone git@github.com:n2b8/birdwatcher.git
cd birdwatcher

# Set up virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

---

## Environment Variables

These should be provided in your systemd `.service` files or exported manually:

```bash
export TELEGRAM_BOT_TOKEN="your-bot-token"
export TELEGRAM_CHAT_ID="your-chat-id"
```

---

## Systemd Services

Enable and manage background tasks:

### 🔍 Detection (YOLOv8 + Hailo)

- `birdwatcher.motion_service.service`: runs `detect_birds_yolo.py`

### 🧠 Classification Queue

- `birdwatcher.classifier_service.service`: runs `classify_queue.py`

### 🌐 Web Dashboard

- `birdwatcher.web_service.service`: runs `app.py` via Flask

```bash
sudo systemctl daemon-reload
sudo systemctl enable birdwatcher.motion_service.service
sudo systemctl enable birdwatcher.classifier_service.service
sudo systemctl enable birdwatcher.web_service.service

sudo systemctl start birdwatcher.motion_service.service
sudo systemctl start birdwatcher.classifier_service.service
sudo systemctl start birdwatcher.web_service.service
```

---

## Web Interface

- `/` – Gallery of accepted visits
- `/review` – Tag or discard uncertain predictions
- `/stats` – Frequency charts + heatmaps

---

## Custom EfficientNet-B7 Classifier

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/YOUR_NOTEBOOK_FILE_ID)

This project includes a custom-trained **EfficientNet-B7 model** built specifically for backyard bird feeder species using a filtered subset of the [NABirds dataset](https://dl.allaboutbirds.org/nabirds). 

- Trained on 600×600 resolution images using mixed precision
- Uses class-balanced sampling to mitigate dataset imbalance
- Achieved **93.14% validation accuracy** with early stopping
- Training was performed on Google Colab (A100 GPU) with full checkpointing and ONNX export support

### Training Curve

![Training Curve](https://raw.githubusercontent.com/n2b8/birdwatcher/main/docs/training_curve.png)

---

## License

GPL License