import os
import random
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
from scipy.ndimage import gaussian_filter


def create_textured_background(width, height, base_color=245, noise_intensity=10):
    """
    Create a textured background with subtle noise.

    Args:
        width: Image width
        height: Image height
        base_color: Base gray value (0-255)
        noise_intensity: Intensity of texture noise

    Returns:
        PIL Image with textured background
    """
    # Create base background
    bg = np.ones((height, width, 3), dtype=np.float32) * base_color

    # Add subtle noise texture
    noise = np.random.normal(0, noise_intensity, (height, width, 3))
    bg = bg + noise

    # Clip to valid range
    bg = np.clip(bg, 0, 255).astype(np.uint8)

    return Image.fromarray(bg)


def create_plain_background(width, height):
    """
    Create a plain gray background.

    Args:
        width: Image width
        height: Image height

    Returns:
        PIL Image with plain background
    """
    # Random gray value between 220-255
    gray_value = random.randint(220, 255)
    bg = np.ones((height, width, 3), dtype=np.uint8) * gray_value

    return Image.fromarray(bg)


def create_grain_mask(height, width, num_spots_range=(10, 20), spot_size_range=(50, 200), blur_sigma=50):
    """
    Create a mask with random spots for grain effect.

    Args:
        height: Image height
        width: Image width
        num_spots_range: Tuple of (min, max) number of spots
        spot_size_range: Tuple of (min, max) radius for spots
        blur_sigma: Sigma for gaussian blur

    Returns:
        Normalized mask array with values between 0 and 1
    """
    mask = np.zeros((height, width), dtype=np.float32)

    num_spots = np.random.randint(num_spots_range[0], num_spots_range[1] + 1)

    for _ in range(num_spots):
        center_y = np.random.randint(0, height)
        center_x = np.random.randint(0, width)
        base_radius = np.random.randint(spot_size_range[0], spot_size_range[1])

        aspect_ratio = np.random.uniform(0.4, 1.5)
        angle = np.random.uniform(0, 2 * np.pi)

        y, x = np.ogrid[:height, :width]
        y_shifted = y - center_y
        x_shifted = x - center_x

        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        x_rot = cos_angle * x_shifted + sin_angle * y_shifted
        y_rot = -sin_angle * x_shifted + cos_angle * y_shifted

        radius_x = base_radius
        radius_y = base_radius * aspect_ratio

        distance = np.sqrt((x_rot / radius_x)**2 + (y_rot / radius_y)**2)

        deformation = np.random.uniform(0.85, 1.15)
        threshold = 1.0 * deformation

        spot_intensity = np.random.uniform(0.3, 1.0)
        spot = (distance <= threshold).astype(np.float32) * spot_intensity

        if np.random.random() > 0.5:
            noise_scale = np.random.uniform(0.05, 0.15)
            spot = spot * (1 - noise_scale + noise_scale * 2 * np.random.random(spot.shape))

        mask = np.maximum(mask, spot)

    mask = gaussian_filter(mask, sigma=blur_sigma)

    if mask.max() > 0:
        mask = mask / mask.max()

    return mask


def apply_grain(image, grain_intensity):
    """
    Apply grain effect to image.

    Args:
        image: PIL Image
        grain_intensity: Intensity of grain (0-100)

    Returns:
        PIL Image with grain applied
    """
    if grain_intensity == 0:
        return image

    img_array = np.array(image).astype(np.float32)
    height, width = img_array.shape[:2]

    mask = create_grain_mask(height, width, num_spots_range=(10, 18),
                            spot_size_range=(80, 250), blur_sigma=60)

    if len(img_array.shape) == 3:
        mask = mask[:, :, np.newaxis]

    noise = np.random.normal(0, grain_intensity, img_array.shape)
    noisy_img = img_array + (noise * mask)
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

    return Image.fromarray(noisy_img)


def load_object_images(objects_dir):
    """
    Load all object images from directory.

    Args:
        objects_dir: Path to directory with object images

    Returns:
        List of PIL Images
    """
    objects = []
    objects_path = Path(objects_dir)

    for img_file in sorted(objects_path.glob("*.jpg")):
        img = Image.open(img_file).convert('RGB')
        objects.append(img)

    print(f"Loaded {len(objects)} object images from {objects_dir}")
    return objects


def generate_synthetic_image(objects, img_size=1280, num_objects_range=(10, 30),
                            scale_range=(0.7, 1.3), use_textured_bg=False):
    """
    Generate a synthetic image with randomly placed objects.

    Args:
        objects: List of PIL Images (object crops)
        img_size: Size of output image (square)
        num_objects_range: Tuple of (min, max) number of objects to place
        scale_range: Tuple of (min, max) scale factor for objects
        use_textured_bg: Whether to use textured background

    Returns:
        Tuple of (PIL Image, list of annotations)
        Annotations format: [(class_id, x_center, y_center, width, height), ...]
        All coordinates are normalized (0-1)
    """
    # Create background
    if use_textured_bg:
        canvas = create_textured_background(img_size, img_size)
    else:
        canvas = create_plain_background(img_size, img_size)

    # Randomly select number of objects
    num_objects = random.randint(num_objects_range[0], num_objects_range[1])

    annotations = []

    for _ in range(num_objects):
        # Randomly select an object
        obj = random.choice(objects)

        # Random scale
        scale = random.uniform(scale_range[0], scale_range[1])
        new_width = int(obj.width * scale)
        new_height = int(obj.height * scale)

        # Resize object
        obj_scaled = obj.resize((new_width, new_height), Image.LANCZOS)

        # Random position (ensure object is fully inside image)
        max_x = img_size - new_width
        max_y = img_size - new_height

        if max_x <= 0 or max_y <= 0:
            continue  # Skip if object is too large

        x = random.randint(0, max_x)
        y = random.randint(0, max_y)

        # Paste object on canvas
        canvas.paste(obj_scaled, (x, y))

        # Calculate YOLO format annotation (normalized coordinates)
        x_center = (x + new_width / 2) / img_size
        y_center = (y + new_height / 2) / img_size
        width = new_width / img_size
        height = new_height / img_size

        # Class ID is 0 (single class)
        annotations.append((0, x_center, y_center, width, height))

    return canvas, annotations


def save_yolo_annotation(annotations, output_path):
    """
    Save annotations in YOLO format.

    Args:
        annotations: List of (class_id, x_center, y_center, width, height)
        output_path: Path to save .txt file
    """
    with open(output_path, 'w') as f:
        for ann in annotations:
            class_id, x_center, y_center, width, height = ann
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def generate_dataset(objects_dir, output_dir, num_train=500, num_val=150, num_test=100,
                     img_size=1280, num_objects_range=(10, 30), scale_range=(0.7, 1.3)):
    """
    Generate complete YOLO dataset with train/val/test splits.

    Args:
        objects_dir: Directory with object crop images
        output_dir: Output directory for dataset
        num_train: Number of training images
        num_val: Number of validation images
        num_test: Number of test images
        img_size: Size of generated images
        num_objects_range: Range of objects per image
        scale_range: Range of object scales
    """
    # Load object images
    objects = load_object_images(objects_dir)

    if len(objects) == 0:
        print("ERROR: No object images found!")
        return

    # Create output directories
    output_path = Path(output_dir)
    for split in ['train', 'val', 'test']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)

    # Grain intensity levels (4 levels)
    grain_levels = [0, 10, 20, 30]

    # Generate datasets
    splits = [
        ('train', num_train),
        ('val', num_val),
        ('test', num_test)
    ]

    total_generated = 0

    for split_name, num_images in splits:
        # Check how many images already exist
        existing_images = list((output_path / split_name / 'images').glob('*.jpg'))
        num_existing = len(existing_images)

        print(f"\n{'='*60}")
        print(f"Generating {split_name} set: {num_images} images")
        print(f"Existing: {num_existing} images")
        print(f"{'='*60}")

        if num_existing >= num_images:
            print(f"[SKIP] {split_name} set already complete ({num_existing}/{num_images})")
            total_generated += num_existing
            continue

        start_idx = num_existing
        if start_idx > 0:
            print(f"[RESUME] Continuing from image {start_idx}")

        for i in range(start_idx, num_images):
            # Decide background type (50/50 split)
            use_textured = random.random() < 0.5
            bg_type = "textured" if use_textured else "plain"

            # Generate synthetic image
            img, annotations = generate_synthetic_image(
                objects=objects,
                img_size=img_size,
                num_objects_range=num_objects_range,
                scale_range=scale_range,
                use_textured_bg=use_textured
            )

            num_objs = len(annotations)

            # Apply random grain level
            grain_intensity = random.choice(grain_levels)
            img = apply_grain(img, grain_intensity)

            # Convert to grayscale
            img = img.convert('L')

            # Save image and annotation
            img_filename = f"{split_name}_{i:05d}.jpg"
            img_path = output_path / split_name / 'images' / img_filename
            ann_path = output_path / split_name / 'labels' / f"{split_name}_{i:05d}.txt"

            img.save(img_path, quality=95)
            save_yolo_annotation(annotations, ann_path)

            total_generated += 1

            # More verbose logging
            progress_pct = ((i + 1) / num_images) * 100
            if (i + 1) % 10 == 0 or (i + 1) <= 5:
                print(f"  [{progress_pct:5.1f}%] {i + 1}/{num_images} | "
                      f"Objs: {num_objs:2d} | BG: {bg_type:8s} | Grain: {grain_intensity:2d}")

        print(f"[OK] {split_name} set completed: {num_images} images")

    print(f"\n{'='*60}")
    print(f"DATASET GENERATION COMPLETE!")
    print(f"{'='*60}")
    print(f"Total images generated: {total_generated}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Configuration
    OBJECTS_DIR = "data/datasets/objects-v2"
    OUTPUT_DIR = "data/yolo_dataset"

    # Generation parameters (as agreed)
    IMG_SIZE = 1280
    NUM_OBJECTS_RANGE = (10, 30)
    SCALE_RANGE = (0.7, 1.3)

    # Dataset sizes
    NUM_TRAIN = 700
    NUM_VAL = 200
    NUM_TEST = 100

    print("="*60)
    print("YOLO DATASET GENERATOR")
    print("="*60)
    print(f"Objects directory: {OBJECTS_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Objects per image: {NUM_OBJECTS_RANGE[0]}-{NUM_OBJECTS_RANGE[1]}")
    print(f"Scale range: {SCALE_RANGE[0]}x-{SCALE_RANGE[1]}x")
    print(f"Train: {NUM_TRAIN}, Val: {NUM_VAL}, Test: {NUM_TEST}")
    print(f"Total: {NUM_TRAIN + NUM_VAL + NUM_TEST} images")
    print("="*60)

    # Generate dataset
    generate_dataset(
        objects_dir=OBJECTS_DIR,
        output_dir=OUTPUT_DIR,
        num_train=NUM_TRAIN,
        num_val=NUM_VAL,
        num_test=NUM_TEST,
        img_size=IMG_SIZE,
        num_objects_range=NUM_OBJECTS_RANGE,
        scale_range=SCALE_RANGE
    )
