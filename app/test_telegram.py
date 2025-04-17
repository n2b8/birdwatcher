import os
from send_telegram import send_telegram_message

# Dummy values to test
test_caption = "Test: A new House Finch visited your feeder!"
test_image_path = "visits/visit_20250405_170454.jpg"  # Use a real image path here

# Ensure environment variables are loaded
if "TELEGRAM_BOT_TOKEN" not in os.environ or "TELEGRAM_CHAT_ID" not in os.environ:
    print("❌ Please set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in your environment.")
else:
    send_telegram_message(test_caption, test_image_path)
