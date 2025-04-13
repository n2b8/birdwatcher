from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from db import (
    get_connection,
    update_status,
    delete_visit,
    get_not_a_bird_count,
    add_visit,
)

app = Flask(__name__)
IMAGE_DIR = "images"
os.makedirs(IMAGE_DIR, exist_ok=True)

def format_species_name(raw_name):
    # Skip already formatted entries
    if "(" in raw_name or not re.match(r"^\d+_", raw_name):
        return raw_name

    # Remove numeric ID
    _, base = raw_name.split("_", 1)
    base = base.replace("_", " ")

    # Try to split the last 2+ words as qualifier if known patterns
    known_qualifiers = [
        "Adult Male", "Adult Female", "Juvenile", "Immature",
        "Female immature", "Female juvenile", "Breeding Audubons", "Nonbreeding"
    ]

    for q in known_qualifiers:
        if base.endswith(q):
            name = base[: -len(q)].strip()
            qualifier = q.replace(" ", "/") if "_" in raw_name else q
            return f"{name} ({qualifier})"

    # Fallback: split last word only
    tokens = base.split()
    if len(tokens) > 1:
        name = " ".join(tokens[:-1])
        qualifier = tokens[-1]
        return f"{name} ({qualifier})"

    return base.strip()

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

    not_a_bird_count = get_not_a_bird_count()
    has_next = (offset + per_page) < total_count
    has_prev = page > 1

    return render_template("index.html", entries=rows, not_a_bird_count=not_a_bird_count,
                           page=page, has_next=has_next, has_prev=has_prev)

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

    # Top 10 bar chart
    top_species = df["species"].value_counts().head(10)
    plt.figure(figsize=(10, 5))
    top_species.plot(kind="barh", color="skyblue")
    plt.xlabel("Number of Visits")
    plt.ylabel("Species")
    plt.title("Top 10 Most Frequently Seen Species")
    plt.tight_layout()
    plt.savefig("static/species_bar.png")
    plt.close()

    # Heatmap
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day_name()
    heatmap_data = df.groupby(["day", "hour"]).size().unstack(fill_value=0)
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
