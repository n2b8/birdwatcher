from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import csv
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import requests
from db import (
    get_connection,
    update_status,
    delete_visit,
    add_visit,
)

app = Flask(__name__, static_folder="static")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
LABEL_MAP_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ai", "model", "label_map.csv")
CLASS_LABEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ai", "model", "class_labels.txt")
THUMBNAIL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "thumbnails")
IMAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "images")
os.makedirs(IMAGE_DIR, exist_ok=True)

# Load species label mapping from CSV
SPECIES_LOOKUP = {}
with open(LABEL_MAP_PATH, newline="", encoding="utf-8") as f:
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

def fetch_current_weather(lat=39.7392, lon=-104.9903, timezone="auto"):  # Default: Denver, CO
    try:
        url = "https://api.open-meteo.com/v1/gfs"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,cloud_cover,precipitation_probability",
            "current": "temperature_2m,cloud_cover,precipitation_probability",
            "temperature_unit": "fahrenheit",
            "precipitation_unit": "inch",
            "timezone": timezone,
        }
        resp = requests.get(url, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        # Fallback to first hourly forecast if 'current' isn't available
        hour_index = 0
        temp = data["hourly"]["temperature_2m"][hour_index]
        cloud = data["hourly"]["cloud_cover"][hour_index]
        precip = data["hourly"]["precipitation_probability"][hour_index]

        icon = "☀️"
        if cloud > 80:
            icon = "☁️"
        elif cloud > 40:
            icon = "⛅"
        if precip > 60:
            icon = "🌧️"

        return f"{icon} {round(temp)}°F"
    except Exception as e:
        print(f"[WARN] Weather fetch failed: {e}")
        return "Weather unavailable"

@app.route("/")
def index():
    page = request.args.get("page", default=1, type=int)
    per_page = 10
    offset = (page - 1) * per_page

    today = datetime.now().date()

    with get_connection() as conn:
        # Main gallery entries
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

        # Total visit count
        cursor = conn.execute("""
            SELECT COUNT(*) FROM visits
            WHERE status = 'accepted'
              AND LOWER(species) != 'not_a_bird'
        """)
        total_count = cursor.fetchone()[0]

        # Today's visit count
        cursor = conn.execute("""
            SELECT COUNT(*) FROM visits
            WHERE status = 'accepted'
              AND DATE(timestamp) = ?
              AND LOWER(species) != 'not_a_bird'
        """, (today,))
        todays_count = cursor.fetchone()[0]

        # Most recent visit today
        cursor = conn.execute("""
            SELECT * FROM visits
            WHERE status = 'accepted'
              AND DATE(timestamp) = ?
              AND LOWER(species) != 'not_a_bird'
            ORDER BY timestamp DESC
            LIMIT 1
        """, (today,))
        recent_row = cursor.fetchone()
        most_recent = dict(zip([col[0] for col in cursor.description], recent_row)) if recent_row else None
        if most_recent:
            most_recent["species"] = format_species_name(most_recent["species"])

        # Most frequent species today
        cursor = conn.execute("""
            SELECT species, COUNT(*) as count
            FROM visits
            WHERE status = 'accepted'
              AND DATE(timestamp) = ?
              AND LOWER(species) != 'not_a_bird'
            GROUP BY species
            ORDER BY count DESC
            LIMIT 1
        """, (today,))
        freq_row = cursor.fetchone()
        most_frequent_species = format_species_name(freq_row[0]) if freq_row else None

    has_next = (offset + per_page) < total_count
    has_prev = page > 1

    return render_template(
        "index.html",
        entries=rows,
        page=page,
        has_next=has_next,
        has_prev=has_prev,
        date=today.strftime("%A, %B %d"),
        todays_count=todays_count,
        most_recent=most_recent,
        most_frequent_species=most_frequent_species,
        weather=fetch_current_weather()
    )

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
            WHERE status = 'accepted'
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
    top_species.plot(kind="barh", color=["#4b6b48" if i % 2 == 0 else "#836953" for i in range(len(top_species))])
    plt.xlabel("Number of Detections")
    plt.ylabel("Species")
    plt.title("Top 10 Most Frequently Detected Species")
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, "species_bar.png"))
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

    # Annotated custom color heatmap
    from matplotlib.colors import LinearSegmentedColormap

    custom_cmap = LinearSegmentedColormap.from_list(
        "woodsy",
        ["#f3f1ed", "#4b6b48", "#3a5a70"]  # light -> green -> slate blue
    )

    plt.figure(figsize=(14, 6))
    sns.heatmap(
        activity,
        cmap=custom_cmap,
        annot=True,
        fmt="d",
        linewidths=0.5,
        cbar_kws={"label": "Detections"}
    )
    plt.title("Bird Detection Density by Time of Day and Weekday")
    plt.xlabel("Hour of Day")
    plt.ylabel("Day of Week")
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, "visit_heatmap.png"))
    plt.close()

    return render_template("stats.html")

@app.route("/images/<path:filename>")
def serve_image(filename):
    return send_from_directory(IMAGE_DIR, filename)

@app.route("/thumbnails/<path:filename>")
def serve_thumbnail(filename):
    return send_from_directory(THUMBNAIL_DIR, filename)

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

@app.route("/edit/<filename>", methods=["GET", "POST"])
def edit_species(filename):
    if request.method == "POST":
        new_species = request.form["species"]
        with get_connection() as conn:
            conn.execute("""
                UPDATE visits
                SET species = ?, confidence = 0.0, classified = 1
                WHERE filename = ?
            """, (new_species, filename))
            conn.commit()
        return redirect(url_for("index"))

    # GET: Show dropdown
    with open(CLASS_LABEL_PATH) as f:
        class_labels = [line.strip() for line in f]

    return render_template("edit.html", filename=filename, species_options=class_labels)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
