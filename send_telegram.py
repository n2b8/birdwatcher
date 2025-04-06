import os
import requests

def send_telegram_message(species, image_path=None):
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not bot_token or not chat_id:
        print("❌ TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set.")
        return

    message = f"A new {species} has visited your feeder! 🐦"

    # Send photo with caption if image is provided
    if image_path and os.path.exists(image_path):
        with open(image_path, "rb") as photo:
            response = requests.post(
                f"https://api.telegram.org/bot{bot_token}/sendPhoto",
                data={"chat_id": chat_id, "caption": message},
                files={"photo": photo}
            )
    else:
        # Just send a text message
        response = requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            data={"chat_id": chat_id, "text": message}
        )

    if response.ok:
        print("✅ Telegram notification sent.")
    else:
        print(f"❌ Telegram failed: {response.text}")

if __name__ == "__main__":
    import sys
    species = sys.argv[1] if len(sys.argv) > 1 else "Unknown Bird"
    image_path = sys.argv[2] if len(sys.argv) > 2 else None
    send_telegram_message(species, image_path)
