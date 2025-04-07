from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import csv
from datetime import datetime
from picamera2 import Picamera2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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
    cleaned = []

    if os.path.exists(REVIEW_CSV):
        with open(REVIEW_CSV, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_path = os.path.join(REVIEW_DIR, row["filename"])
                if os.path.exists(image_path):
                    entries.append(row)
                else:
                    cleaned.append(row["filename"])

        if cleaned:
            print(f"[CLEANUP] Removing missing images from review_log.csv: {cleaned}")
            with open(REVIEW_CSV, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["filename", "species", "confidence", "timestamp"])
                writer.writeheader()
                for row in entries:
                    writer.writerow({
                        "filename": row.get("filename", ""),
                        "species": row.get("species", ""),
                        "confidence": row.get("confidence", ""),
                        "timestamp": row.get("timestamp", "")
                    })

    print(f"[REVIEW] Loaded {len(entries)} entries, cleaned {len(cleaned)}")
    return render_template("review.html", entries=entries)

@app.route("/stats")
def stats():
    if not os.path.exists(LOG_CSV):
        return "No data available yet."

    df = pd.read_csv(LOG_CSV, parse_dates=["timestamp"])

    # Species frequency bar chart
    top_species = df["species"].value_counts().head(10)
    plt.figure(figsize=(10, 5))
    top_species.plot(kind="barh", color="skyblue")
    plt.xlabel("Number of Visits")
    plt.ylabel("Species")
    plt.title("Top 10 Most Frequently Seen Species")
    plt.tight_layout()
    plt.savefig("static/species_bar.png")
    plt.close()

    # Heatmap: visits by hour and day of week
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day_name()
    heatmap_data = df.groupby(["day", "hour"]).size().unstack(fill_value=0)

    # Ensure correct day order
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heatmap_data = heatmap_data.reindex(day_order)

    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_data, cmap="YlGnBu")
    plt.title("Bird Visit Frequency by Time of Day and Weekday")
    plt.xlabel("Hour of Day")
    plt.ylabel("Day of Week")
    plt.tight_layout()
    plt.savefig("static/visit_heatmap.png")
    plt.close()

    return render_template("stats.html")

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

    if os.path.exists(REVIEW_CSV):
        with open(REVIEW_CSV, "r") as f:
            lines = list(csv.DictReader(f))

        with open(REVIEW_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "species", "confidence", "timestamp"])
            writer.writeheader()
            for row in lines:
                if row["filename"] != filename:
                    writer.writerow({
                        "filename": row.get("filename", ""),
                        "species": row.get("species", ""),
                        "confidence": row.get("confidence", ""),
                        "timestamp": row.get("timestamp", "")
                    })
                else:
                    log_row = {
                        "filename": row.get("filename", ""),
                        "species": row.get("species", ""),
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    write_header = not os.path.exists(LOG_CSV) or os.stat(LOG_CSV).st_size == 0
                    with open(LOG_CSV, "a", newline="") as f_log:
                        log_writer = csv.DictWriter(f_log, fieldnames=["filename", "species", "timestamp"])
                        if write_header:
                            log_writer.writeheader()
                        log_writer.writerow(log_row)

    return redirect(url_for("review"))

@app.route("/mark_not_a_bird/<filename>", methods=["POST"])
def mark_not_a_bird(filename):
    review_path = os.path.join(REVIEW_DIR, filename)
    discard_path = os.path.join("not_a_bird", filename)
    os.makedirs("not_a_bird", exist_ok=True)

    # Move image if it exists
    if os.path.exists(review_path):
        os.rename(review_path, discard_path)

    # Update review_log.csv
    if os.path.exists(REVIEW_CSV):
        with open(REVIEW_CSV, "r") as f:
            lines = list(csv.DictReader(f))

        new_lines = [row for row in lines if row.get("filename") != filename]

        with open(REVIEW_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "species", "confidence", "timestamp"])
            writer.writeheader()
            for row in new_lines:
                writer.writerow({
                    "filename": row.get("filename", ""),
                    "species": row.get("species", ""),
                    "confidence": row.get("confidence", ""),
                    "timestamp": row.get("timestamp", "")
                })

    return redirect(url_for("review"))

@app.route("/snap")
def snap():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"visit_{timestamp}.jpg"
    filepath = os.path.join("visits", filename)

    result = os.system(f"libcamera-still -n --width 640 --height 480 -o {filepath}")
    if result != 0 or not os.path.exists(filepath):
        return "❌ Failed to capture image", 500

    with open(LOG_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if os.stat(LOG_CSV).st_size == 0:
            writer.writerow(["filename", "species", "timestamp"])
        writer.writerow([filename, "Manual Snapshot", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
