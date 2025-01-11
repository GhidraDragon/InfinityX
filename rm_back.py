#!/usr/bin/env python3

import sys
import os
from rembg import remove
from PIL import Image

def main():
    # Check for correct usage
    if len(sys.argv) != 2:
        print("Usage: python remove_bg.py <image_path>")
        sys.exit(1)
    
    # Get the image path from the command line argument
    image_path = sys.argv[1]
    
    # Verify the file actually exists
    if not os.path.isfile(image_path):
        print(f"Error: File not found: {image_path}")
        sys.exit(1)
    
    # Perform background removal
    try:
        with open(image_path, "rb") as input_file:
            input_image = Image.open(input_file)
            output_image = remove(input_image)  # Remove background using rembg
            
            # Build output filename
            file_root, file_ext = os.path.splitext(image_path)
            output_path = f"{file_root}_nobg.png"
            
            # Save the resulting image
            output_image.save(output_path)
            print(f"Background removed. Output saved to: {output_path}")
    except Exception as e:
        print(f"An error occurred while processing {image_path}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()