import os
import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter


def create_grain_mask(height, width, num_spots_range=(10, 20), spot_size_range=(50, 200), blur_sigma=50):
    """
    Create a mask with random spots that have gradient edges and organic shapes.

    Args:
        height: Image height
        width: Image width
        num_spots_range: Tuple of (min, max) number of spots
        spot_size_range: Tuple of (min, max) radius for spots
        blur_sigma: Sigma for gaussian blur to create gradient edges

    Returns:
        Normalized mask array with values between 0 and 1
    """
    mask = np.zeros((height, width), dtype=np.float32)

    # Random number of spots
    num_spots = np.random.randint(num_spots_range[0], num_spots_range[1] + 1)

    # Generate random spots with varied shapes
    for _ in range(num_spots):
        # Random center position
        center_y = np.random.randint(0, height)
        center_x = np.random.randint(0, width)

        # Random size
        base_radius = np.random.randint(spot_size_range[0], spot_size_range[1])

        # Random ellipse parameters for organic shape
        # Aspect ratio: how stretched the ellipse is
        aspect_ratio = np.random.uniform(0.4, 1.5)

        # Random rotation angle
        angle = np.random.uniform(0, 2 * np.pi)

        # Create coordinate grid
        y, x = np.ogrid[:height, :width]
        y_shifted = y - center_y
        x_shifted = x - center_x

        # Apply rotation
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        x_rot = cos_angle * x_shifted + sin_angle * y_shifted
        y_rot = -sin_angle * x_shifted + cos_angle * y_shifted

        # Create elliptical distance with aspect ratio
        radius_x = base_radius
        radius_y = base_radius * aspect_ratio

        # Elliptical distance
        distance = np.sqrt((x_rot / radius_x)**2 + (y_rot / radius_y)**2)

        # Add some organic deformation using noise
        deformation = np.random.uniform(0.85, 1.15)
        threshold = 1.0 * deformation

        # Add spot with intensity variation
        spot_intensity = np.random.uniform(0.3, 1.0)
        spot = (distance <= threshold).astype(np.float32) * spot_intensity

        # Optionally add some perlin-like noise for more organic edges
        if np.random.random() > 0.5:
            noise_scale = np.random.uniform(0.05, 0.15)
            spot = spot * (1 - noise_scale + noise_scale * 2 * np.random.random(spot.shape))

        mask = np.maximum(mask, spot)

    # Apply gaussian blur for gradient edges
    mask = gaussian_filter(mask, sigma=blur_sigma)

    # Normalize to [0, 1] range
    if mask.max() > 0:
        mask = mask / mask.max()

    return mask


def add_grain(image_path, output_path, grain_intensity=30, num_spots_range=(10, 20), spot_size_range=(50, 200), blur_sigma=50):
    """
    Add grain/noise effect to an image in spotted pattern with gradient edges and random organic shapes.

    Args:
        image_path: Path to input image
        output_path: Path to save the processed image
        grain_intensity: Intensity of grain effect (0-100, default 30)
        num_spots_range: Tuple of (min, max) number of grainy spots
        spot_size_range: Tuple of (min, max) radius for spots
        blur_sigma: Sigma for gradient edge blur (higher = softer edges)
    """
    # Load image
    img = Image.open(image_path)
    img_array = np.array(img).astype(np.float32)

    height, width = img_array.shape[:2]

    # Create grain mask with gradient spots of random shapes
    mask = create_grain_mask(height, width, num_spots_range, spot_size_range, blur_sigma)

    # Expand mask to match image channels if needed
    if len(img_array.shape) == 3:
        mask = mask[:, :, np.newaxis]

    # Generate grain noise
    noise = np.random.normal(0, grain_intensity, img_array.shape)

    # Apply noise only where mask indicates (with gradient transition)
    noisy_img = img_array + (noise * mask)

    # Clip values to valid range [0, 255]
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

    # Convert back to PIL Image
    result = Image.fromarray(noisy_img)

    # Convert to grayscale (black and white)
    result = result.convert('L')

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save result
    result.save(output_path, quality=95)


if __name__ == "__main__":
    from datetime import datetime
    import glob

    # Input directory
    input_dir = r"data\datasets\objects-v2"

    # Get all jpg files
    image_files = glob.glob(os.path.join(input_dir, "*.jpg"))

    if not image_files:
        print(f"No images found in {input_dir}")
        exit(1)

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", f"dataset-{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Parameters
    num_intensity_levels = 20  # Количество уровней интенсивности
    variants_per_intensity = 7  # Вариантов для каждой интенсивности
    min_intensity = 0
    max_intensity = 25

    print(f"Processing {len(image_files)} images from {input_dir}")
    print(f"Intensity levels: {num_intensity_levels} (from {min_intensity} to {max_intensity})")
    print(f"Variants per intensity: {variants_per_intensity}")
    print(f"Total images to generate: {len(image_files) * num_intensity_levels * variants_per_intensity}")
    print(f"Output directory: {output_dir}\n")

    total_count = 0

    # Process each image
    for img_idx, input_file in enumerate(sorted(image_files), 1):
        filename = os.path.basename(input_file)
        base_name, ext = os.path.splitext(filename)

        print(f"[{img_idx}/{len(image_files)}] Processing: {filename}")

        # Create images for each intensity level
        for intensity_idx in range(num_intensity_levels):
            # Calculate intensity for this level
            intensity = min_intensity + (max_intensity - min_intensity) * (intensity_idx / (num_intensity_levels - 1))

            # Create multiple variants for this intensity
            for variant_idx in range(1, variants_per_intensity + 1):
                # Create unique filename
                output_filename = f"{base_name}_int{int(intensity):02d}_var{variant_idx:02d}{ext}"
                output_file = os.path.join(output_dir, output_filename)

                # Add grain effect with random spotted pattern
                add_grain(
                    input_file,
                    output_file,
                    grain_intensity=intensity,
                    num_spots_range=(10, 18),
                    spot_size_range=(80, 250),
                    blur_sigma=60
                )
                total_count += 1

            print(f"  Intensity {int(intensity):2d}: {variants_per_intensity} variants created")

        print()

    print(f"\n{'='*60}")
    print(f"ALL DONE!")
    print(f"{'='*60}")
    print(f"Total images processed: {len(image_files)}")
    print(f"Total images generated: {total_count}")
    print(f"Output location: {output_dir}")
    print(f"{'='*60}")
