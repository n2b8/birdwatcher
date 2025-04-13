from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import csv
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from db import (
    get_connection,
    update_status,
    delete_visit,
    add_visit,
)

app = Flask(__name__)
IMAGE_DIR = "images"
os.makedirs(IMAGE_DIR, exist_ok=True)

# Load species label mapping from CSV
SPECIES_LOOKUP = {}
with open("model/label_map.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        species_id = row["ID"].strip()
        name = row["Species"].strip()
        subtitle = row["Subtitle"].strip() if "Subtitle" in row and row["Subtitle"] else None
        SPECIES_LOOKUP[species_id] = (name, subtitle)

def format_species_name(raw_name):
    if not raw_name:
        return "Unknown"

    raw_name = raw_name.strip()

    # Handle special case
    if raw_name.lower() == "not_a_bird":
        return "Not a Bird"

    # Match leading numeric ID
    match = re.match(r"^(\d+)", raw_name)
    if match:
        species_id = match.group(1)
        if species_id in SPECIES_LOOKUP:
            name, subtitle = SPECIES_LOOKUP[species_id]
            return f"{name} ({subtitle})" if subtitle else name

    return raw_name  # fallback if no match

@app.route("/")
def index():
    page = request.args.get("page", default=1, type=int)
    per_page = 10
    offset = (page - 1) * per_page

    with get_connection() as conn:
        cursor = conn.execute("""
            SELECT * FROM visits
            WHERE status = 'accepted'
              AND LOWER(species) != 'not_a_bird'
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
        """, (per_page, offset))
        rows = [dict(zip([col[0] for col in cursor.description], row)) for row in cursor.fetchall()]
        for row in rows:
            row["species"] = format_species_name(row["species"])

        cursor = conn.execute("""
            SELECT COUNT(*) FROM visits
            WHERE status = 'accepted'
              AND LOWER(species) != 'not_a_bird'
        """)
        total_count = cursor.fetchone()[0]

    has_next = (offset + per_page) < total_count
    has_prev = page > 1

    return render_template("index.html", entries=rows, page=page, has_next=has_next, has_prev=has_prev)

@app.route("/review")
def review():
    page = request.args.get("page", default=1, type=int)
    per_page = 10
    offset = (page - 1) * per_page

    with get_connection() as conn:
        cursor = conn.execute("""
            SELECT * FROM visits
            WHERE status IN ('review', 'not_a_bird') AND classified = 1
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
        """, (per_page, offset))
        rows = [dict(zip([col[0] for col in cursor.description], row)) for row in cursor.fetchall()]
        for row in rows:
            row["species"] = format_species_name(row["species"])

        cursor = conn.execute("""
            SELECT COUNT(*) FROM visits
            WHERE status IN ('review', 'not_a_bird') AND classified = 1
        """)
        total_count = cursor.fetchone()[0]

    has_next = (offset + per_page) < total_count
    has_prev = page > 1

    return render_template("review.html", entries=rows, page=page, has_next=has_next, has_prev=has_prev)

@app.route("/stats")
def stats():
    with get_connection() as conn:
        cursor = conn.execute("""
            SELECT * FROM visits
            WHERE confidence IS NOT NULL AND confidence >= 0.7
        """)
        rows = [dict(zip([col[0] for col in cursor.description], row)) for row in cursor.fetchall()]

    if not rows:
        return "No data available yet."

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # 👇 Apply standardized species formatting
    df["species"] = df["species"].apply(format_species_name)

    # Top 10 bar chart
    top_species = df["species"].value_counts().head(10)
    plt.figure(figsize=(10, 5))
    top_species.plot(kind="barh", color="skyblue")
    plt.xlabel("Number of Detections")
    plt.ylabel("Species")
    plt.title("Top 10 Most Frequently Detected Species")
    plt.tight_layout()
    plt.savefig("static/species_bar.png")
    plt.close()

    # Heatmap: Bird detection density
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day_name()

    all_hours = list(range(24))
    all_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    # Create a full 7x24 grid
    full_index = pd.MultiIndex.from_product([all_days, all_hours], names=["day", "hour"])
    activity = df.groupby(["day", "hour"]).size().reindex(full_index, fill_value=0).unstack()
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    activity = activity.reindex(day_order)  # ✅ force correct row order

    # Annotated single-hue heatmap
    plt.figure(figsize=(14, 6))
    sns.heatmap(
        activity,
        cmap="Blues",
        annot=True,
        fmt="d",
        linewidths=0.5,
        cbar_kws={"label": "Detections"}
    )
    plt.title("Bird Detection Density by Time of Day and Weekday")
    plt.xlabel("Hour of Day")
    plt.ylabel("Day of Week")
    plt.tight_layout()
    plt.savefig("static/visit_heatmap.png")
    plt.close()

    return render_template("stats.html")

@app.route("/images/<path:filename>")
def serve_image(filename):
    return send_from_directory(IMAGE_DIR, filename)

@app.route("/thumbnails/<path:filename>")
def serve_thumbnail(filename):
    return send_from_directory("thumbnails", filename)

@app.route("/mark_good/<filename>", methods=["POST"])
def mark_good(filename):
    update_status(filename, "accepted")
    return redirect(url_for("review"))

@app.route("/mark_not_a_bird/<filename>", methods=["POST"])
def mark_not_a_bird(filename):
    update_status(filename, "not_a_bird")
    return redirect(url_for("review"))

@app.route("/delete/<filename>", methods=["POST"])
def delete(filename):
    image_path = os.path.join(IMAGE_DIR, filename)
    if os.path.exists(image_path):
        os.remove(image_path)
    delete_visit(filename)

    # Redirect based on where the request came from
    referer = request.headers.get("Referer", "")
    if "/review" in referer:
        return redirect(url_for("review"))
    return redirect(url_for("index"))

@app.route("/snap")
def snap():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"visit_{timestamp}.jpg"
    temp_path = os.path.join("/tmp", filename)
    final_path = os.path.join(IMAGE_DIR, filename)

    result = os.system(f"libcamera-still -n --width 640 --height 480 -o {temp_path}")
    if result != 0 or not os.path.exists(temp_path):
        return "❌ Failed to capture image", 500

    os.rename(temp_path, final_path)

    # Save to DB
    add_visit(
        filename=filename,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        species="Manual Snapshot",
        confidence=None,
        motion_score=None,
        status="accepted"
    )

    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
