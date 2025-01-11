#!/usr/bin/env python3

import sys
import os
import uuid
import datetime

from PIL import Image
import numpy as np

try:
    # For demonstration: We'll use scikit-learn's random distributions to apply small noise
    from sklearn.utils import check_random_state
except ImportError:
    raise ImportError("Please install scikit-learn (pip install scikit-learn) to use the random noise demonstration.")

def add_images_auto_resize_mlaware(image_path_1, image_path_2, output_path):
    """
    Adds two images together after resizing both to 1024x1024 pixels,
    then applies a simple "machine learning" based random color shift or noise.
    Result is saved as a PNG.

    :param image_path_1: Path to the first image file
    :param image_path_2: Path to the second image file
    :param output_path: Path to save the resulting PNG
    """
    # 1. Load both images
    img1 = Image.open(image_path_1).convert("RGBA")
    img2 = Image.open(image_path_2).convert("RGBA")

    # 2. Resize both images to 1024x1024
    #    Use LANCZOS if your Pillow version supports it; fallback to ANTIALIAS otherwise.
    #    LANCZOS is recommended for high-quality downsampling.
    if hasattr(Image, 'Resampling'):
        img1 = img1.resize((1024, 1024), Image.Resampling.LANCZOS)
        img2 = img2.resize((1024, 1024), Image.Resampling.LANCZOS)
    else:
        img1 = img1.resize((1024, 1024), Image.ANTIALIAS)
        img2 = img2.resize((1024, 1024), Image.ANTIALIAS)

    # 3. Convert images to NumPy arrays
    arr1 = np.array(img1, dtype=np.uint8)
    arr2 = np.array(img2, dtype=np.uint8)

    # 4. Add the images element-wise, clip to [0, 255]
    added_array = arr1.astype(np.int32) + arr2.astype(np.int32)
    added_array = np.clip(added_array, 0, 255).astype(np.uint8)

    # 5. Apply a simple random color shift / random noise (machine learning demonstration)
    #    We'll do a random shift in the RGBA channels for demonstration—not a real copyright-avoidance method.
    random_state = check_random_state(None)  # or seed with an integer if you want reproducibility
    #   Let's add some Gaussian noise in each channel
    noise = random_state.normal(loc=0, scale=10, size=added_array.shape)  # small noise with std=10
    noisy_array = added_array.astype(float) + noise
    #   Clip back to valid range
    noisy_array = np.clip(noisy_array, 0, 255).astype(np.uint8)

    # 6. Convert back to a PIL Image
    final_image = Image.fromarray(noisy_array, mode="RGBA")

    # 7. Save the result as PNG
    final_image.save(output_path, format="PNG")


def create_intelligent_output_filename(img1_path, img2_path, prefix="merged_"):
    """
    Create a somewhat "intelligent" output filename by combining parts of the input filenames,
    plus a timestamp and a random uuid to ensure uniqueness.
    """
    # Extract base names (without extension)
    base1 = os.path.splitext(os.path.basename(img1_path))[0]
    base2 = os.path.splitext(os.path.basename(img2_path))[0]

    # Current date/time
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Random UUID
    random_str = str(uuid.uuid4())[:8]

    # Combine
    filename = f"{prefix}{base1}_{base2}_{now_str}_{random_str}.png"
    return filename


def main():
    """
    Main function: 
      usage: python script.py <image1> <image2>
      Output path is automatically generated.
    """
    if len(sys.argv) < 3:
        print("Usage: python script.py <image1> <image2>")
        sys.exit(1)

    image_path_1 = sys.argv[1]
    image_path_2 = sys.argv[2]

    output_file = create_intelligent_output_filename(image_path_1, image_path_2)
    add_images_auto_resize_mlaware(image_path_1, image_path_2, output_file)

    print(f"Output image saved as '{output_file}'")

if __name__ == "__main__":
    main()