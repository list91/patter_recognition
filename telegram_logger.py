"""
Telegram Logger for YOLO Training
Sends training progress graphs to Telegram after each epoch
"""

import os
import csv
import requests
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN = "8494675923:AAFpnSkEC8QhyexRzzW9i-714o2oJmHAicE"
TELEGRAM_CHAT_ID = "1425764006"
GRAPH_PATH = "runs/detect/scheme_detector/mAP_progress.png"


def read_results_csv(csv_path):
    """
    Read results.csv and extract epochs and mAP50-95 values

    Returns:
        tuple: (epochs list, mAP50-95 list) or (None, None) if error
    """
    if not os.path.exists(csv_path):
        print(f"[TG Logger] CSV file not found: {csv_path}")
        return None, None

    epochs = []
    map_values = []

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                epoch = float(row['epoch'])
                map_value = float(row['metrics/mAP50-95(B)'])
                epochs.append(epoch)
                map_values.append(map_value)

        return epochs, map_values
    except Exception as e:
        print(f"[TG Logger] Error reading CSV: {e}")
        return None, None


def render_graph(epochs, map_values, output_path):
    """
    Render mAP50-95 progress graph

    Args:
        epochs: List of epoch numbers
        map_values: List of mAP50-95 values
        output_path: Path to save the graph image

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create figure with high DPI for better quality
        plt.figure(figsize=(12, 6), dpi=100)

        # Plot the data
        plt.plot(epochs, map_values, 'b-', linewidth=2, label='mAP50-95')
        plt.scatter(epochs, map_values, c='blue', s=30, alpha=0.6)

        # Highlight the best value
        best_idx = map_values.index(max(map_values))
        plt.scatter([epochs[best_idx]], [map_values[best_idx]],
                   c='red', s=100, marker='*', zorder=5,
                   label=f'Best: {map_values[best_idx]:.4f}')

        # Styling
        plt.xlabel('Epoch', fontsize=12, fontweight='bold')
        plt.ylabel('mAP50-95', fontsize=12, fontweight='bold')
        plt.title('YOLO Training Progress - mAP50-95 over Epochs',
                 fontsize=14, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(loc='best', fontsize=10)

        # Add statistics text box
        stats_text = f'Current: {map_values[-1]:.4f}\n'
        stats_text += f'Best: {max(map_values):.4f}\n'
        stats_text += f'Epochs: {len(epochs)}'
        plt.text(0.02, 0.98, stats_text,
                transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save the figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()

        print(f"[TG Logger] Graph saved to: {output_path}")
        return True

    except Exception as e:
        print(f"[TG Logger] Error rendering graph: {e}")
        plt.close()
        return False


def send_telegram_message(text):
    """
    Send text message to Telegram

    Args:
        text: Message text

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": "HTML"
        }
        response = requests.post(url, json=data, timeout=10)

        if response.status_code == 200:
            print(f"[TG Logger] Message sent successfully")
            return True
        else:
            print(f"[TG Logger] Failed to send message: {response.status_code}")
            return False

    except Exception as e:
        print(f"[TG Logger] Error sending message: {e}")
        return False


def send_telegram_photo(photo_path, caption):
    """
    Send photo to Telegram

    Args:
        photo_path: Path to the image file
        caption: Photo caption

    Returns:
        bool: True if successful, False otherwise
    """
    if not os.path.exists(photo_path):
        print(f"[TG Logger] Photo file not found: {photo_path}")
        return False

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"

        with open(photo_path, 'rb') as photo_file:
            files = {'photo': photo_file}
            data = {
                'chat_id': TELEGRAM_CHAT_ID,
                'caption': caption,
                'parse_mode': 'HTML'
            }
            response = requests.post(url, files=files, data=data, timeout=30)

        if response.status_code == 200:
            print(f"[TG Logger] Photo sent successfully")
            return True
        else:
            print(f"[TG Logger] Failed to send photo: {response.status_code}")
            print(f"[TG Logger] Response: {response.text}")
            return False

    except Exception as e:
        print(f"[TG Logger] Error sending photo: {e}")
        return False


def on_epoch_end(trainer):
    """
    Callback function to be called at the end of each training epoch

    Args:
        trainer: YOLO trainer object
    """
    try:
        print("\n" + "="*60)
        print("[TG Logger] Processing epoch completion...")
        print("="*60)

        # Get the CSV path from trainer
        csv_path = Path(trainer.save_dir) / "results.csv"

        if not csv_path.exists():
            print(f"[TG Logger] Results CSV not found yet: {csv_path}")
            return

        # Read the CSV data
        epochs, map_values = read_results_csv(str(csv_path))

        if epochs is None or len(epochs) == 0:
            print("[TG Logger] No data to plot yet")
            return

        # Render the graph
        graph_path = Path(trainer.save_dir) / "mAP_progress.png"
        if not render_graph(epochs, map_values, str(graph_path)):
            print("[TG Logger] Failed to render graph")
            return

        # Prepare the message
        current_epoch = int(epochs[-1])
        current_map = map_values[-1]
        best_map = max(map_values)
        best_epoch = int(epochs[map_values.index(best_map)])

        caption = f"<b>📊 Training Progress Update</b>\n\n"
        caption += f"<b>Current Epoch:</b> {current_epoch}\n"
        caption += f"<b>Current mAP50-95:</b> {current_map:.4f}\n"
        caption += f"<b>Best mAP50-95:</b> {best_map:.4f} (epoch {best_epoch})\n"
        caption += f"<b>Total Epochs:</b> {len(epochs)}\n"

        # Calculate improvement
        if len(map_values) > 1:
            improvement = current_map - map_values[-2]
            if improvement > 0:
                caption += f"<b>Change:</b> +{improvement:.4f} ⬆️\n"
            elif improvement < 0:
                caption += f"<b>Change:</b> {improvement:.4f} ⬇️\n"
            else:
                caption += f"<b>Change:</b> No change ➡️\n"

        # Send to Telegram
        send_telegram_photo(str(graph_path), caption)

        print("="*60)
        print("[TG Logger] Epoch processing complete")
        print("="*60 + "\n")

    except Exception as e:
        print(f"[TG Logger] Error in on_epoch_end: {e}")
        import traceback
        traceback.print_exc()


# For standalone testing
if __name__ == "__main__":
    print("Testing Telegram Logger...")

    # Test with existing data
    csv_path = "runs/detect/scheme_detector/results.csv"

    if os.path.exists(csv_path):
        print("\n1. Reading CSV...")
        epochs, map_values = read_results_csv(csv_path)

        if epochs:
            print(f"   Found {len(epochs)} epochs")
            print(f"   Latest mAP50-95: {map_values[-1]:.4f}")

            print("\n2. Rendering graph...")
            graph_path = "runs/detect/scheme_detector/mAP_progress_test.png"
            if render_graph(epochs, map_values, graph_path):
                print(f"   Graph saved successfully")

                print("\n3. Sending to Telegram...")
                caption = f"<b>🧪 Test Message</b>\n\n"
                caption += f"<b>Latest Epoch:</b> {int(epochs[-1])}\n"
                caption += f"<b>Latest mAP50-95:</b> {map_values[-1]:.4f}\n"
                caption += f"<b>Best mAP50-95:</b> {max(map_values):.4f}"

                if send_telegram_photo(graph_path, caption):
                    print("   [OK] Test successful!")
                else:
                    print("   [ERROR] Failed to send to Telegram")
            else:
                print("   [ERROR] Failed to render graph")
        else:
            print("   [ERROR] Failed to read CSV data")
    else:
        print(f"   [ERROR] CSV file not found: {csv_path}")
        print("\n   Testing with message only...")
        send_telegram_message("🧪 <b>Test message from YOLO Telegram Logger</b>")
