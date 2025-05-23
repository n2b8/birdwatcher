import time
import os
import sys
import subprocess
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "app"))
from db import get_connection, delete_visit

CLASSIFY_INTERVAL = 60  # seconds

def get_oldest_unclassified():
    with get_connection() as conn:
        c = conn.cursor()
        c.execute("""
            SELECT filename FROM visits
            WHERE classified = 0
            ORDER BY timestamp ASC
            LIMIT 1
        """)
        row = c.fetchone()
        return row[0] if row else None

def mark_classified(filename):
    with get_connection() as conn:
        c = conn.cursor()
        c.execute("UPDATE visits SET classified = 1 WHERE filename = ?", (filename,))
        conn.commit()

def classify_image(filename):
    path = os.path.join("images", filename)
    if not os.path.exists(path):
        print(f"[WARN] Image {filename} not found.")
        mark_classified(filename)  # skip it
        return

    print(f"[CLASSIFY] Processing {filename}")
    result = subprocess.run([
        "python3", "/home/jake/birdwatcher/ai/classify_bird.py",
        path,
        filename
    ])

    if result.returncode == 2:  # special return code for discarded
        print(f"[CLEANUP] '{filename}' was not a bird — removing visit entry.")
        delete_visit(filename)
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
    else:
        mark_classified(filename)

def classify_loop():
    print("[INFO] Starting classification queue loop...")
    while True:
        image = get_oldest_unclassified()
        if image:
            classify_image(image)
        else:
            print("[INFO] No unclassified images found.")
        time.sleep(CLASSIFY_INTERVAL)

if __name__ == "__main__":
    classify_loop()
