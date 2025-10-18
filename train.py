"""
YOLO Model Training Script
Optimized for small object detection on scheme/diagram images
"""

from ultralytics import YOLO
import torch

def main():
    print("="*60)
    print("YOLO TRAINING SCRIPT")
    print("="*60)

    # Check GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print("="*60)

    # Check if we have a saved model to resume from
    import os
    last_model_path = 'runs/detect/scheme_detector/weights/last.pt'
    resume_training = False

    if os.path.exists(last_model_path):
        print(f"\n[FOUND] Previous training checkpoint: {last_model_path}")
        print("Resuming training from last checkpoint...")
        model = YOLO(last_model_path)
        resume_training = True
    else:
        print("\n[NEW] No previous checkpoint found")
        print("Starting fresh training with pretrained YOLOv8n...")
        # Load YOLOv8 model (nano version for faster training)
        # You can change to: yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt for better accuracy
        model = YOLO('yolov8n.pt')

    print("\nTraining configuration:")
    print("Dataset: data/yolo_dataset/dataset.yaml")
    print("Model: YOLOv8n (nano)")
    print(f"Resume: {resume_training}")
    print("\nTraining parameters:")
    print("  - Image size: 1280x1280")
    print("  - Batch size: 8")
    print("  - Epochs: 100")
    print("  - Workers: 4")
    print("  - Device:", device)
    print("="*60 + "\n")

    # Train the model
    # Parameters optimized for small objects:
    # - imgsz=1280: Higher resolution helps detect small objects
    # - batch=8: Adjust based on GPU memory
    # - epochs=100: Can increase for better results
    # - patience=20: Early stopping if no improvement
    # - resume=True: Continue from last checkpoint if available
    results = model.train(
        data='data/yolo_dataset/dataset.yaml',
        epochs=100,
        imgsz=1280,
        batch=8,
        device=device,
        workers=4,
        patience=20,
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        project='runs/detect',
        name='scheme_detector',
        exist_ok=True,
        resume=resume_training,  # Resume if checkpoint found
        pretrained=True,
        optimizer='AdamW',
        verbose=True,
        seed=42,
        deterministic=True,
        single_cls=True,  # Single class detection
        rect=False,  # Rectangular training
        cos_lr=True,  # Cosine learning rate scheduler
        close_mosaic=10,  # Disable mosaic augmentation in last 10 epochs
        amp=True,  # Automatic Mixed Precision
        fraction=1.0,  # Use 100% of dataset
        # Augmentation parameters
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,  # No rotation (as per your requirement)
        translate=0.1,
        scale=0.2,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,  # No vertical flip
        fliplr=0.0,  # No horizontal flip
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
    )

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Best model saved at: runs/detect/scheme_detector/weights/best.pt")
    print(f"Last model saved at: runs/detect/scheme_detector/weights/last.pt")
    print("="*60)

    # Validate the model
    print("\nRunning validation...")
    metrics = model.val()

    print("\n" + "="*60)
    print("VALIDATION METRICS")
    print("="*60)
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
