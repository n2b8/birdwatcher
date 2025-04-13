# Birdwatcher 🐦📸

> A full-stack AI bird identification system deployed on Raspberry Pi, combining real-time object detection, custom fine-grained classification, web-based analytics, and cloud backup — all running on hardware the size of a credit card.

---

## Features

- 🦾 **YOLOv8 (Hailo)** real-time bird detection using `rpicam-hello`
- 📷 High-quality still capture on detection
- 🧠 Custom EfficientNet-B7 classification (ONNX model, trained on a subset of the [NABirds dataset](https://dl.allaboutbirds.org/nabirds))
- 🗂️ Images + metadata stored in SQLite (`visits` table)
- 🔍 Review interface for uncertain predictions
- 📊 Species frequency charts + time-based heatmap
- 📡 Telegram notifications on high-confidence visits
- ☁️ Daily S3 backups of visits + metadata
- 🌐 Web server for dashboard, review, and analytics
- 🛠️ systemd services for background classification & detection

---

## System Architecture

1. **YOLOv8 (Hailo-accelerated)** detects birds in real-time via `rpicam-hello`
2. Captured frames are passed to a **custom EfficientNet-B7 classifier**
3. Predictions are stored in a **SQLite database** with image metadata
4. Confirmed images are saved to disk and displayed on a **Flask-based web dashboard**
5. **Daily S3 backups** archive visit logs and images
6. Optional **Telegram notifications** on high-confidence predictions

---

## Custom EfficientNet-B7 Classifier

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/YOUR_NOTEBOOK_FILE_ID)

Trained from scratch using a feeder-bird-only subset of the NABirds dataset:

- Input resolution: `600×600`
- Class-balanced sampling
- Mixed precision (AMP)
- Early stopping and resume support
- Best validation accuracy: **93.14%** (Epoch 23)
- Final model exported as ONNX

### Training Curve

![Training Curve](https://raw.githubusercontent.com/n2b8/birdwatcher/main/docs/training_curve.png)

### Model Artifacts
```
model/
├── class_labels_v3.txt
└── efficientnet_b7_backyard_feeder_birds.onnx
```

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

## Model Hosting & Reusability

The custom EfficientNet-B7 model will be hosted for public use:

- [ ] Hugging Face Model Card (planned)
- [ ] Kaggle Model Dataset (optional)
- [ ] GitHub LFS or Releases (fallback)

---

## Demo & Media

Coming soon:

- Hardware photos
- Screencasts of the web UI
- Live bird detection and classification demos

---

## License

GPL License