import os
import csv
import subprocess

VISITS_DIR = "visits"
LOG_FILE = os.path.join(VISITS_DIR, "log.csv")
REVIEW_LOG = os.path.join("review", "review_log.csv")

# Gather filenames already logged
logged_files = set()

for log_path in [LOG_FILE, REVIEW_LOG]:
    if os.path.exists(log_path):
        with open(log_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                logged_files.add(row["filename"])

# Find orphaned images
all_images = [f for f in os.listdir(VISITS_DIR) if f.endswith(".jpg")]
orphans = [f for f in all_images if f not in logged_files]

print(f"Found {len(orphans)} orphaned images")

# Re-run classify_bird.py
for image in orphans:
    print(f"🔍 Rechecking: {image}")
    subprocess.run(["python3", "classify_bird.py", os.path.join(VISITS_DIR, image)])
