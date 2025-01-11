#!/usr/bin/env python3

import argparse
import os
import sys
import datetime
from io import BytesIO
from PIL import Image, ImageFilter, ImageEnhance, ImageOps

def compress_jpeg_to_target_size(img, desired_size_bytes, min_quality=1, max_quality=95):
    """
    Attempt to compress the image to be <= desired_size_bytes using a binary search on quality.
    Returns (compressed_data, quality) if successful, or (None, None) if not.
    """
    best_quality = None
    best_data = None
    
    # Convert/ensure mode is RGB for JPEG
    if img.mode not in ["RGB", "RGBA"]:
        img = img.convert("RGB")
    if img.mode == "RGBA":
        # Flatten alpha channel onto white background or similar
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])  # 3 is the alpha channel
        img = background

    while min_quality <= max_quality:
        mid_quality = (min_quality + max_quality) // 2
        
        buffer = BytesIO()
        try:
            # We force the format to JPEG here
            img.save(buffer, format="JPEG", quality=mid_quality)
        except Exception:
            # If something goes wrong, break out
            break
        
        size = buffer.tell()

        if size <= desired_size_bytes:
            # We can try to increase the quality
            best_quality = mid_quality
            best_data = buffer.getvalue()
            min_quality = mid_quality + 1
        else:
            # We need to decrease the quality
            max_quality = mid_quality - 1

    return best_data, best_quality

def main():
    parser = argparse.ArgumentParser(
        description='A script that applies a wide range of photo editing tools to an image.'
    )
    parser.add_argument('input_image', type=str, 
                        help='Path to the input image.')
    parser.add_argument('--sharpness', type=float, default=1.0,
                        help='Adjust the sharpness factor (1.0 = no change). Higher = sharper, lower = blurrier.')
    parser.add_argument('--brightness', type=float, default=1.0,
                        help='Adjust the brightness factor (1.0 = no change). Higher = brighter, lower = darker.')
    parser.add_argument('--contrast', type=float, default=1.0,
                        help='Adjust the contrast factor (1.0 = no change). Higher = more contrast, lower = less contrast.')
    parser.add_argument('--color', type=float, default=1.0,
                        help='Adjust the color factor (1.0 = no change). Higher = more vivid, lower = grayscale.')
    parser.add_argument('--grayscale', action='store_true',
                        help='Convert the image to grayscale.')
    parser.add_argument('--invert', action='store_true',
                        help='Invert the colors of the image.')
    parser.add_argument('--flip', choices=['horizontal', 'vertical'],
                        help='Flip the image horizontally or vertically.')
    parser.add_argument('--rotate', type=int,
                        help='Rotate the image by the specified number of degrees.')
    parser.add_argument('--blur', action='store_true',
                        help='Apply a blur filter to the image.')
    parser.add_argument('--contour', action='store_true',
                        help='Apply a contour filter to the image.')
    parser.add_argument('--edges', action='store_true',
                        help='Apply an edge-enhancement filter to the image.')
    parser.add_argument('--emboss', action='store_true',
                        help='Apply an emboss filter to the image.')
    parser.add_argument('--find_edges', action='store_true',
                        help='Apply a find-edges filter to the image.')
    parser.add_argument('--smooth', action='store_true',
                        help='Apply a smoothing filter to the image.')
    parser.add_argument('--sharpen_filter', action='store_true',
                        help='Apply a sharpen filter to the image.')
    parser.add_argument('--autocontrast', action='store_true',
                        help='Automatically maximize contrast of the image.')
    parser.add_argument('--equalize', action='store_true',
                        help='Equalize the image histogram.')
    parser.add_argument('--posterize', type=int,
                        help='Posterize the image using the given number of bits (1-8).')
    parser.add_argument('--solarize', type=int,
                        help='Solarize the image using the given threshold (0-255).')
    parser.add_argument('--resize_dims', type=int, nargs=2, default=None, metavar=('WIDTH', 'HEIGHT'),
                        help='Resize image to (WIDTH x HEIGHT) in pixels.')
    parser.add_argument('--resize_fs_percent', type=int, default=None,
                        help='Resize/compress to the given percentage of the original file size (1-100). '
                             'This uses JPEG compression; if the input is not JPEG, the output will be forced to JPEG.')
    parser.add_argument('--output', type=str,
                        help='Output filename (or path). If not specified, a “smart” filename will be generated.',
                        default=None)

    args = parser.parse_args()

    # Attempt to open the image
    try:
        img = Image.open(args.input_image)
    except Exception as e:
        print(f"Error: Unable to open image '{args.input_image}'. Reason: {e}")
        sys.exit(1)

    # =======================================================
    # 1. Optional explicit (width, height) resize if provided
    # =======================================================
    if args.resize_dims is not None:
        width, height = args.resize_dims
        if width <= 0 or height <= 0:
            print("Error: Width and Height must be positive.")
            sys.exit(1)
        img = img.resize((width, height), resample=Image.LANCZOS)

    # =======================================
    # 2. Apply transformations as in original
    # =======================================

    # Adjust sharpness
    if args.sharpness != 1.0:
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(args.sharpness)

    # Adjust brightness
    if args.brightness != 1.0:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(args.brightness)

    # Adjust contrast
    if args.contrast != 1.0:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(args.contrast)

    # Adjust color saturation
    if args.color != 1.0:
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(args.color)

    # Convert to grayscale
    if args.grayscale:
        img = ImageOps.grayscale(img)

    # Invert colors (ensure image mode is RGB or RGBA first)
    if args.invert:
        if img.mode == 'RGBA':
            r, g, b, a = img.split()
            rgb = Image.merge('RGB', (r, g, b))
            inv_rgb = ImageOps.invert(rgb)
            r2, g2, b2 = inv_rgb.split()
            img = Image.merge('RGBA', (r2, g2, b2, a))
        else:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = ImageOps.invert(img)

    # Flip image
    if args.flip == 'horizontal':
        img = ImageOps.mirror(img)
    elif args.flip == 'vertical':
        img = ImageOps.flip(img)

    # Rotate image
    if args.rotate:
        img = img.rotate(args.rotate, expand=True)

    # Apply filters
    if args.blur:
        img = img.filter(ImageFilter.BLUR)
    if args.contour:
        img = img.filter(ImageFilter.CONTOUR)
    if args.edges:
        img = img.filter(ImageFilter.EDGE_ENHANCE)
    if args.emboss:
        img = img.filter(ImageFilter.EMBOSS)
    if args.find_edges:
        img = img.filter(ImageFilter.FIND_EDGES)
    if args.smooth:
        img = img.filter(ImageFilter.SMOOTH)
    if args.sharpen_filter:
        img = img.filter(ImageFilter.SHARPEN)

    # Apply additional transformations
    if args.autocontrast:
        img = ImageOps.autocontrast(img)
    if args.equalize:
        img = ImageOps.equalize(img)
    if args.posterize is not None:
        bits = args.posterize
        if bits < 1 or bits > 8:
            print("Error: Posterize bits must be between 1 and 8.")
            sys.exit(1)
        img = ImageOps.posterize(img, bits)
    if args.solarize is not None:
        threshold = args.solarize
        if threshold < 0 or threshold > 255:
            print("Error: Solarize threshold must be between 0 and 255.")
            sys.exit(1)
        img = ImageOps.solarize(img, threshold=threshold)

    # =========================================================
    # 3. Determine output filename (if not provided) + Generate
    # =========================================================
    if not args.output:
        file_root, file_ext = os.path.splitext(args.input_image)
        transformations = []

        # Gather transformations for the “smart” suffix
        if args.resize_dims is not None:
            transformations.append(f"resize-{args.resize_dims[0]}x{args.resize_dims[1]}")
        if args.resize_fs_percent is not None:
            transformations.append(f"resizefs-{args.resize_fs_percent}pct")
        if args.sharpness != 1.0:
            transformations.append(f"sharp-{args.sharpness}")
        if args.brightness != 1.0:
            transformations.append(f"bright-{args.brightness}")
        if args.contrast != 1.0:
            transformations.append(f"contrast-{args.contrast}")
        if args.color != 1.0:
            transformations.append(f"color-{args.color}")
        if args.grayscale:
            transformations.append("grayscale")
        if args.invert:
            transformations.append("invert")
        if args.flip:
            transformations.append(f"flip-{args.flip}")
        if args.rotate:
            transformations.append(f"rotate-{args.rotate}")
        if args.blur:
            transformations.append("blur")
        if args.contour:
            transformations.append("contour")
        if args.edges:
            transformations.append("edges")
        if args.emboss:
            transformations.append("emboss")
        if args.find_edges:
            transformations.append("find-edges")
        if args.smooth:
            transformations.append("smooth")
        if args.sharpen_filter:
            transformations.append("sharpen-filter")
        if args.autocontrast:
            transformations.append("autocontrast")
        if args.equalize:
            transformations.append("equalize")
        if args.posterize is not None:
            transformations.append(f"posterize-{args.posterize}")
        if args.solarize is not None:
            transformations.append(f"solarize-{args.solarize}")

        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        transformations_str = "_".join(transformations)

        if transformations_str:
            smart_filename = f"{file_root}_{transformations_str}_{timestamp}{file_ext}"
        else:
            # No transformations were performed
            smart_filename = f"{file_root}_edited_{timestamp}{file_ext}"

        args.output = smart_filename

    # Ensure output directory exists (if any)
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            print(f"Error creating directory '{output_dir}': {e}")
            sys.exit(1)

    # ===================================================================
    # 4. (Optional) Resize to a percentage of original file size (JPEG-only)
    # ===================================================================
    if args.resize_fs_percent is not None:
        # Check percentage bounds
        if args.resize_fs_percent < 1 or args.resize_fs_percent > 100:
            print("Error: --resize_fs_percent must be between 1 and 100.")
            sys.exit(1)

        original_size = None
        try:
            original_size = os.path.getsize(args.input_image)
        except Exception as e:
            print(f"Warning: Could not get size of '{args.input_image}': {e}")
            print("Will attempt resizing by file size anyway...")
        
        if original_size:
            desired_size_bytes = int((args.resize_fs_percent / 100.0) * original_size)
            compressed_data, quality = compress_jpeg_to_target_size(img, desired_size_bytes)
            if compressed_data is None:
                # Could not compress to desired size
                print("Warning: Could not compress image to the desired file size. Saving with default settings.")
                try:
                    img.save(args.output)
                    print(f"Image saved to '{args.output}'.")
                except Exception as e:
                    print(f"Error: Unable to save image to '{args.output}'. Reason: {e}")
                    sys.exit(1)
            else:
                # Successfully found a quality level
                # Write compressed_data to file
                # Force .jpg extension if current output does not match
                _, ext = os.path.splitext(args.output)
                if ext.lower() not in ['.jpg', '.jpeg']:
                    args.output = args.output + '.jpg'

                try:
                    with open(args.output, 'wb') as f:
                        f.write(compressed_data)
                    print(f"Image saved to '{args.output}' at ~{args.resize_fs_percent}% of original size (quality={quality}).")
                except Exception as e:
                    print(f"Error: Unable to save image to '{args.output}'. Reason: {e}")
                    sys.exit(1)
        else:
            # If we couldn't determine the original size, just do a normal save
            print("Warning: Could not determine original file size. Saving normally.")
            try:
                img.save(args.output)
                print(f"Image saved to '{args.output}'.")
            except Exception as e:
                print(f"Error: Unable to save image to '{args.output}'. Reason: {e}")
                sys.exit(1)
    else:
        # If not resizing by file size, just save the image normally in the original (or current) format
        try:
            img.save(args.output)
            print(f"Image saved to '{args.output}'.")
        except Exception as e:
            print(f"Error: Unable to save image to '{args.output}'. Reason: {e}")
            sys.exit(1)

if __name__ == '__main__':
    main()