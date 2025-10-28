"""
YOLO Model Training Script - Low Memory Configuration
Optimized for servers with limited RAM (3-4 GB)
"""

from ultralytics import YOLO
import torch
import gc
import telegram_logger

def main():
    print("="*60)
    print("YOLO TRAINING SCRIPT - LOW MEMORY MODE")
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
        model = YOLO('yolov8n.pt')

    # Clear memory
    gc.collect()

    # Register Telegram Logger callback
    print("\n[TELEGRAM LOGGER] Registering callback for epoch updates...")
    model.add_callback("on_train_epoch_end", telegram_logger.on_epoch_end)
    print("[TELEGRAM LOGGER] Callback registered successfully!")

    print("\nTraining configuration (LOW MEMORY):")
    print("Dataset: data/yolo_dataset/dataset.yaml")
    print("Model: YOLOv8n (nano)")
    print(f"Resume: {resume_training}")
    print("\nOptimized parameters for 3.7GB RAM:")
    print("  - Image size: 1280x1280 (full resolution)")
    print("  - Batch size: 2 (reduced from 8)")
    print("  - Epochs: 1000")
    print("  - Workers: 1 (reduced from 4)")
    print("  - Cache: False (to save memory)")
    print("  - Dataset: 100% (1200 images)")
    print("  - Device:", device)
    print("  - Telegram Updates: ENABLED")
    print("="*60 + "\n")

    # Train the model with LOW MEMORY settings
    results = model.train(
        data='data/yolo_dataset/dataset.yaml',
        epochs=1000,
        imgsz=1280,  # FULL RESOLUTION: 1280
        batch=2,  # REDUCED from 8 - critical for memory
        device=device,
        workers=1,  # REDUCED from 4 - less memory usage
        patience=20,
        save=True,
        save_period=10,
        project='runs/detect',
        name='scheme_detector',
        exist_ok=True,
        resume=resume_training,  # Will resume from last.pt if exists
        pretrained=True,
        optimizer='AdamW',
        verbose=True,
        seed=42,
        deterministic=True,
        single_cls=True,
        rect=False,
        cos_lr=True,
        close_mosaic=10,
        amp=False,  # Disabled on CPU
        fraction=1.0,  # Use 100% of training dataset (all 1200 images)
        cache=False,  # IMPORTANT: Don't cache to save memory
        # Augmentation parameters
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.2,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,
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
