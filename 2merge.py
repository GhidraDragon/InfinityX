#!/usr/bin/env python3

"""
merge.py

Blends two images' "features" using a simple PCA approach. Instead of saving
to a separate file, this version overwrites <image1> by merging <image2> into it.

Usage:
  python merge.py <image1> <image2> [alpha]

Where:
  - image1, image2 are paths to the two images (PNG, JPG, etc.).
  - alpha (optional float in [0,1]) indicates how heavily image2's
    PCA representation influences the final result. Default = 0.5.

After running, <image1> will be overwritten with the new blended result.
"""

import sys
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize
from sklearn.decomposition import PCA

def load_and_preprocess(img_path, size=(256, 256)):
    """
    Load an image, ensure it has 3 channels (RGB), resize to `size`,
    and return both the (H,W,3) image and the flattened 1D float array.
    
    :param img_path: Path to the input image file
    :param size: Desired (height, width) for the image
    :return: (resized_image, flattened_image_vector)
    """
    # Read the raw image
    image = imread(img_path)
    
    # 1) Ensure 3 channels:
    #    - If grayscale, replicate channels
    #    - If 4 channels (e.g. RGBA), discard alpha
    if image.ndim == 2:
        # shape = (H,W) => grayscale, replicate 3 times
        image_rgb = np.stack([image]*3, axis=-1)
    else:
        channels = image.shape[-1]
        if channels == 3:
            image_rgb = image
        elif channels > 3:
            # E.g., RGBA => discard alpha
            image_rgb = image[..., :3]
        else:
            # If channels < 3 => replicate last channel
            needed = 3 - channels
            image_rgb = np.concatenate([image] + [image[..., -1:]]*needed, axis=-1)
    
    # 2) Resize to (size[0], size[1], 3), ensuring float in [0,1]
    #    Larger sizes give better detail but use more memory/compute.
    image_resized = resize(image_rgb, size, anti_aliasing=True)
    
    # 3) Flatten => (size[0]*size[1]*3,)
    image_flat = image_resized.flatten()
    
    return image_resized, image_flat

def main():
    # Parse arguments
    if len(sys.argv) < 3:
        print("\nUsage: python merge.py <image1> <image2> [alpha]\n")
        print("Example: python merge.py picA.jpg picB.png 0.7\n")
        sys.exit(1)
    
    image1_path = sys.argv[1]
    image2_path = sys.argv[2]
    
    # Default alpha = 0.5 if not provided
    alpha = 0.5
    if len(sys.argv) >= 4:
        try:
            alpha_in = float(sys.argv[3])
            alpha = max(0.0, min(1.0, alpha_in))  # clamp to [0,1]
        except ValueError:
            pass  # If invalid, keep alpha=0.5
    
    # Load and flatten both images => (H,W,3), 1D array
    img1_resized, fv1 = load_and_preprocess(image1_path, size=(256, 256))
    img2_resized, fv2 = load_and_preprocess(image2_path, size=(256, 256))
    
    # Debug prints
    print(f"DEBUG: img1_resized.shape = {img1_resized.shape}")
    print(f"DEBUG: fv1.shape = {fv1.shape}")
    print(f"DEBUG: img2_resized.shape = {img2_resized.shape}")
    print(f"DEBUG: fv2.shape = {fv2.shape}")
    
    # Stack => shape = (2, 256*256*3)
    stacked = np.vstack([fv1, fv2])
    print(f"DEBUG: stacked.shape = {stacked.shape}")
    
    # n_samples = 2, n_features = 256*256*3 => up to 2 PCA components
    n_components = min(2, stacked.shape[0], stacked.shape[1])
    pca = PCA(n_components=n_components)
    
    # Fit/transform => shape (2, n_components)
    pca_data = pca.fit_transform(stacked)
    print(f"DEBUG: pca_data.shape = {pca_data.shape}")
    
    # Blend in PCA space
    blended_pca = (1.0 - alpha)*pca_data[0] + alpha*pca_data[1]
    print(f"DEBUG: blended_pca.shape = {blended_pca.shape}")
    
    # Inverse => returns flat shape (256*256*3)
    blended_flat = pca.inverse_transform(blended_pca)
    print(f"DEBUG: blended_flat.shape = {blended_flat.shape}")
    
    expected_size = 256 * 256 * 3
    if blended_flat.size != expected_size:
        print("\nERROR: Unexpected size for blended_flat. Should be {}.".format(expected_size))
        print("       Something is off with your PCA or shapes.\n")
        return
    
    # Reshape => (256,256,3)
    blended_img = blended_flat.reshape((256, 256, 3))
    print(f"DEBUG: blended_img.shape = {blended_img.shape}")
    
    # Convert to [0,1], then to uint8 for saving
    blended_img = np.clip(blended_img, 0, 1)
    blended_img = (blended_img * 255).astype(np.uint8)
    
    # Save output, overwriting image1
    imsave(image1_path, blended_img)
    
    print(f"\nSuccess! {image1_path} has been overwritten with the blended image. (alpha={alpha})\n")

if __name__ == "__main__":
    main()