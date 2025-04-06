# Birdwatcher 🐦📸

A Raspberry Pi-powered bird identification system that detects motion, captures images, classifies bird species using a custom-trained EfficientNet model, and logs visits through a simple web interface.

---

## Features

- 📷 Motion-triggered image capture using `libcamera-still`
- 🧠 ONNX-based bird species classification (NABirds dataset)
- 🐦 Review interface for low-confidence predictions
- 🚫 "Not a Bird" tagging for model improvement
- 📊 Species frequency stats and visit heatmaps
- 📡 Telegram notifications for each bird visit

---

## Project Structure

```
.
├── app.py                   # Flask web dashboard
├── classify_bird.py         # ONNX bird classification script
├── detect_motion.py         # Motion detection & capture
├── reclassify_visits.py     # Batch reclassification script
├── send_telegram.py         # Notification helper
├── test_telegram.py         # Test script for Telegram bot
├── model/
│   ├── class_labels.txt
│   └── efficientnet_b0_nabirds.onnx
├── visits/                  # Accepted bird photos + log.csv
├── review/                  # Medium-confidence images + review_log.csv
├── not_a_bird/              # Rejected/tagged images for retraining
├── static/                  # Static assets (CSS, symbolic link to visits/)
├── templates/
│   ├── index.html           # Gallery dashboard
│   ├── review.html          # Review UI
│   └── stats.html           # Stats & charts
└── .gitignore
```

---

## Setup

```bash
# Clone the repo
$ git clone git@github.com:n2b8/birdwatcher.git
$ cd birdwatcher

# Set up virtual environment
$ python3 -m venv birdenv
$ source birdenv/bin/activate

# Install dependencies
$ pip install -r requirements.txt

# Set environment variables
$ export TELEGRAM_BOT_TOKEN="your-token"
$ export TELEGRAM_CHAT_ID="your-chat-id"
```

---

## Systemd Services

- `birdwatcher-motion.service`: runs `detect_motion.py`
- `birdwatcher-web.service`: starts the Flask web server

```bash
# Restart services
$ sudo systemctl restart birdwatcher-motion.service
$ sudo systemctl restart birdwatcher-web.service
```

---

## Retraining

Use the images inside `not_a_bird/` to eventually expand your model with a "Not a Bird" class.

---

## Future Features

- 📈 Live charts (Plotly?)
- 📁 Export log.csv
- 🧪 Simple classifier retrain helper

---

## License

GPL License
