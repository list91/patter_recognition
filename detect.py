"""
YOLO Detection Script
Detect scheme elements on test images using trained model
"""

import os
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
import cv2

def main():
    print("="*60)
    print("YOLO DETECTION SCRIPT")
    print("="*60)

    # Paths
    MODEL_PATH = "runs/detect/scheme_detector/weights/best.pt"
    TEST_IMAGES_DIR = "data/test_images"
    OUTPUT_DIR = "results/detections"

    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"\nERROR: Model not found at {MODEL_PATH}")
        print("Please train the model first using: python train.py")
        print("="*60)
        return

    # Check if test images directory exists
    if not os.path.exists(TEST_IMAGES_DIR):
        print(f"\nERROR: Test images directory not found: {TEST_IMAGES_DIR}")
        print("="*60)
        return

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(OUTPUT_DIR) / f"detection_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nModel: {MODEL_PATH}")
    print(f"Test images: {TEST_IMAGES_DIR}")
    print(f"Output directory: {output_path}")
    print("="*60)

    # Load model
    print("\nLoading model...")
    model = YOLO(MODEL_PATH)
    print("Model loaded successfully!")

    # Get all image files
    test_images_path = Path(TEST_IMAGES_DIR)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(test_images_path.glob(f"*{ext}")))
        image_files.extend(list(test_images_path.glob(f"*{ext.upper()}")))

    if len(image_files) == 0:
        print(f"\nERROR: No images found in {TEST_IMAGES_DIR}")
        print("="*60)
        return

    print(f"\nFound {len(image_files)} test images")
    print("="*60)

    # Run detection on each image
    print("\nRunning detection...\n")

    total_detections = 0

    for i, img_file in enumerate(sorted(image_files), 1):
        print(f"[{i}/{len(image_files)}] Processing: {img_file.name}")

        # Run inference
        results = model(str(img_file), conf=0.6, iou=0.45, imgsz=1280, verbose=False)

        # Get the result for the first (and only) image
        result = results[0]

        # Count detections
        num_detections = len(result.boxes)
        total_detections += num_detections

        # Load original image
        img = cv2.imread(str(img_file))

        # Draw bounding boxes
        for box in result.boxes:
            # Get box coordinates (xyxy format)
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Get confidence
            conf = float(box.conf[0])

            # Get class (should be 0 - scheme_element)
            cls = int(box.cls[0])

            # Draw rectangle
            color = (0, 255, 0)  # Green
            thickness = 2
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

            # Draw label with confidence
            label = f"scheme_element: {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2

            # Get text size for background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, font_thickness
            )

            # Draw background rectangle for text
            cv2.rectangle(
                img,
                (x1, y1 - text_height - 10),
                (x1 + text_width, y1),
                color,
                -1  # Filled
            )

            # Draw text
            cv2.putText(
                img,
                label,
                (x1, y1 - 5),
                font,
                font_scale,
                (0, 0, 0),  # Black text
                font_thickness
            )

        # Save annotated image
        output_filename = f"{img_file.stem}_detected{img_file.suffix}"
        output_file = output_path / output_filename
        cv2.imwrite(str(output_file), img)

        print(f"  Detected {num_detections} objects -> Saved to {output_filename}")

    print("\n" + "="*60)
    print("DETECTION COMPLETE!")
    print("="*60)
    print(f"Total images processed: {len(image_files)}")
    print(f"Total objects detected: {total_detections}")
    print(f"Average objects per image: {total_detections / len(image_files):.1f}")
    print(f"Results saved to: {output_path}")
    print("="*60)


if __name__ == "__main__":
    main()
