#!/usr/bin/env python3
"""
Combine two images side by side into one output image,
where the final image has the same size (width and height)
as the first input image (argv[1]). We also use scikit-learn
to generate an "intelligent" output filename by examining
similarities in the base names of the two input files.

Usage:
  python3 combine_side_by_side.py <image1> <image2>

Example:
  python3 combine_side_by_side.py kali_dragon.jpg red_dragon_logo.png

Note:
  - This script resizes both images so they fit side by side
    into the width of the first image (argv[1]) while maintaining
    the height of the first image.
  - It uses PIL (Pillow) for image manipulation and scikit-learn
    purely to create a "more intelligent" output file name
    (based on string similarity of the filenames).
  - Various safety checks are included to help ensure robust usage.
"""

import sys
import os
from PIL import Image

# scikit-learn for "intelligent" filename creation
# (pip install scikit-learn if missing)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def intelligent_output_name(path1, path2):
    """
    Use scikit-learn (Tfidf) to generate an "intelligent" output name
    based on similarity of the filenames (excluding extensions).

    :param path1: str, file path to the first image
    :param path2: str, file path to the second image
    :return: str, proposed output filename (with .jpg extension)
    """
    base1 = os.path.splitext(os.path.basename(path1))[0]
    base2 = os.path.splitext(os.path.basename(path2))[0]

    # Vectorize the two base names
    vectorizer = TfidfVectorizer()
    docs = [base1, base2]
    tfidf_matrix = vectorizer.fit_transform(docs)

    # Calculate similarity
    sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]

    # Generate a combined name reflecting the similarity
    # Example: 'kali_dragon_AND_red_dragon_logo_SIM0.12.jpg'
    combined_name = f"{base1}_AND_{base2}_SIM{sim:.2f}.jpg"
    return combined_name

def combine_images_side_by_side(image_path_1, image_path_2, output_path):
    """
    Combine two images side by side into one output image, ensuring
    the final image has the same dimension as the first image.

    The first half (left side) will contain the first image (resized),
    and the second half (right side) will contain the second image (also resized).

    :param image_path_1: str, path to the first image
    :param image_path_2: str, path to the second image
    :param output_path:  str, desired path for saving the combined image
    """
    # Safety checks for file existence
    if not os.path.isfile(image_path_1):
        raise FileNotFoundError(f"First image not found: {image_path_1}")
    if not os.path.isfile(image_path_2):
        raise FileNotFoundError(f"Second image not found: {image_path_2}")

    # Attempt to open images
    try:
        img1 = Image.open(image_path_1)
    except Exception as e:
        raise OSError(f"Could not open first image: {image_path_1}, error: {e}")
    try:
        img2 = Image.open(image_path_2)
    except Exception as e:
        raise OSError(f"Could not open second image: {image_path_2}, error: {e}")

    # The output image should have the same width and height as the first image
    new_width = img1.width
    new_height = img1.height

    # Ensure we can split the width in two halves
    # If the first image is extremely narrow, this might be impossible
    if new_width < 2:
        raise ValueError("The width of the first image is too small to create a side-by-side output.")

    # Compute left/right widths
    left_half_width = new_width // 2
    right_half_width = new_width - left_half_width  # handles odd widths gracefully

    # Resize both images to fit side-by-side within the first image's dimensions
    # Both will share the same final height (that of the first image)
    img1_resized = img1.resize((left_half_width, new_height), Image.ANTIALIAS)
    img2_resized = img2.resize((right_half_width, new_height), Image.ANTIALIAS)

    # Create a new canvas with the size of the first image
    combined_img = Image.new('RGB', (new_width, new_height), color=(0, 0, 0))

    # Paste images (side by side)
    combined_img.paste(img1_resized, (0, 0))
    combined_img.paste(img2_resized, (left_half_width, 0))

    # Save the result
    combined_img.save(output_path)
    print(f"[INFO] Combined image saved at: {output_path}")

def main():
    """
    Main entry point: parse arguments, generate an intelligent output name using sklearn,
    and create the side-by-side image.
    """
    if len(sys.argv) < 3:
        print("Usage: python3 combine_side_by_side.py <image1> <image2>")
        sys.exit(1)

    image1_path = sys.argv[1]
    image2_path = sys.argv[2]

    # Create an "intelligent" output name
    output_file_path = intelligent_output_name(image1_path, image2_path)

    # Combine and save
    try:
        combine_images_side_by_side(image1_path, image2_path, output_file_path)
    except Exception as err:
        print(f"[ERROR] {err}")
        sys.exit(1)

    print("[INFO] Done.")

if __name__ == "__main__":
    main()