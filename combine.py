#!/usr/bin/env python3

from PIL import Image

def combine_images_side_by_side(image_path_1, image_path_2, output_path):
    """
    Combine two images side by side into one output image.

    :param image_path_1: str, path to the first image
    :param image_path_2: str, path to the second image
    :param output_path:  str, desired path for saving the combined image
    """
    # Open images
    img1 = Image.open(image_path_1)
    img2 = Image.open(image_path_2)

    # Determine combined size
    new_width = img1.width + img2.width
    new_height = max(img1.height, img2.height)

    # Create a new black background canvas
    combined_img = Image.new('RGB', (new_width, new_height), color=(0, 0, 0))

    # Paste images onto the new canvas
    combined_img.paste(img1, (0, 0))
    combined_img.paste(img2, (img1.width, 0))

    # Save the result
    combined_img.save(output_path)
    print(f"Combined image saved at: {output_path}")


if __name__ == "__main__":
    # Example usage:
    # Replace these paths with the actual file paths on your system
    image1_path = "kali_dragon.jpg"       # Path to the white/black Kali dragon
    image2_path = "red_dragon_logo.png"   # Path to the red dragon logo
    output_file_path = "combined_dragons.jpg"

    combine_images_side_by_side(image1_path, image2_path, output_file_path)