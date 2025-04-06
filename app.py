from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import csv
from datetime import datetime
from picamera2 import Picamera2
from PIL import Image

app = Flask(__name__)
VISITS_DIR = "visits"
REVIEW_DIR = "review"
LOG_CSV = os.path.join(VISITS_DIR, "log.csv")
REVIEW_CSV = os.path.join(REVIEW_DIR, "review_log.csv")

# Ensure directories exist
os.makedirs(VISITS_DIR, exist_ok=True)
os.makedirs(REVIEW_DIR, exist_ok=True)

@app.route("/")
def index():
    entries = []
    if os.path.exists(LOG_CSV):
        with open(LOG_CSV, "r") as f:
            reader = csv.DictReader(f)
            entries = sorted(reader, key=lambda x: x["timestamp"], reverse=True)
    return render_template("index.html", entries=entries)

@app.route("/review")
def review():
    entries = []
    if os.path.exists(REVIEW_CSV):
        with open(REVIEW_CSV, "r") as f:
            reader = csv.DictReader(f)
            entries = sorted(reader, key=lambda x: x["timestamp"], reverse=True)
    return render_template("review.html", entries=entries)

@app.route("/visits/<path:filename>")
def serve_image(filename):
    return send_from_directory(VISITS_DIR, filename)

@app.route("/review/<path:filename>")
def serve_review_image(filename):
    return send_from_directory(REVIEW_DIR, filename)

@app.route("/delete/<filename>", methods=["POST"])
def delete(filename):
    filepath = os.path.join(VISITS_DIR, filename)
    if os.path.exists(filepath):
        os.remove(filepath)

    if os.path.exists(LOG_CSV):
        with open(LOG_CSV, "r") as f:
            lines = list(csv.DictReader(f))
        with open(LOG_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "species", "timestamp"])
            writer.writeheader()
            for row in lines:
                if row["filename"] != filename:
                    writer.writerow(row)
    return redirect(url_for("index"))

@app.route("/mark_good/<filename>", methods=["POST"])
def mark_good(filename):
    review_path = os.path.join(REVIEW_DIR, filename)
    final_path = os.path.join(VISITS_DIR, filename)
    if os.path.exists(review_path):
        os.rename(review_path, final_path)

    with open(REVIEW_CSV, "r") as f:
        lines = list(csv.DictReader(f))
    with open(REVIEW_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "species", "timestamp"])
        writer.writeheader()
        for row in lines:
            if row["filename"] != filename:
                writer.writerow(row)

    with open(LOG_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([filename, "Manual Tag", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

    return redirect(url_for("review"))

@app.route("/mark_not_a_bird/<filename>", methods=["POST"])
def mark_not_a_bird(filename):
    review_path = os.path.join(REVIEW_DIR, filename)
    discard_path = os.path.join("not_a_bird", filename)
    os.makedirs("not_a_bird", exist_ok=True)
    if os.path.exists(review_path):
        os.rename(review_path, discard_path)

    with open(REVIEW_CSV, "r") as f:
        lines = list(csv.DictReader(f))
    with open(REVIEW_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "species", "timestamp"])
        writer.writeheader()
        for row in lines:
            if row["filename"] != filename:
                writer.writerow(row)

    return redirect(url_for("review"))

@app.route("/snap")
def snap():
    filename = f"visit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    filepath = os.path.join(VISITS_DIR, filename)

    picam2 = Picamera2()
    config = picam2.create_still_configuration()
    picam2.configure(config)
    picam2.start()
    picam2.capture_file(filepath)
    picam2.stop()

    with open(LOG_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if os.stat(LOG_CSV).st_size == 0:
            writer.writerow(["filename", "species", "timestamp"])
        writer.writerow([filename, "Manual Snapshot", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)