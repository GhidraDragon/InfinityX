#!/usr/bin/env python3
"""
auto_crop_enhance_detect.py
---------------------------
A robust script that:
1. Reads an input image from argv[1].
2. Auto-crops the image by finding the largest bounding contour.
3. Enhances the image (brightness, contrast, optional sharpening).
4. Detects edges using Canny as a demonstration of detection.
5. Saves each step's output to disk for review.

Usage:
    python auto_crop_enhance_detect.py <input_image_path>
"""

import sys
import os
import cv2
import numpy as np

def main():
    # ----------------------------------------------------
    # 0. Validate and parse arguments
    # ----------------------------------------------------
    if len(sys.argv) < 2:
        print("[Error] No image path provided. Usage: python auto_crop_enhance_detect.py <input_image>")
        sys.exit(1)
        
    input_image_path = sys.argv[1]
    
    # Check that the file exists
    if not os.path.isfile(input_image_path):
        print(f"[Error] File not found: {input_image_path}")
        sys.exit(1)
    
    # ----------------------------------------------------
    # 1. Read image from disk
    # ----------------------------------------------------
    # Note: Reading images from untrusted sources can pose security risks if the library
    #       has vulnerabilities. Always keep libraries up-to-date and sanitize inputs.
    try:
        original_img = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
        if original_img is None:
            raise ValueError("OpenCV returned None - possibly an unreadable or corrupt file.")
        print(f"[Info] Successfully loaded image: {input_image_path}")
    except Exception as e:
        print(f"[Error] Failed to read the image. Details: {e}")
        sys.exit(1)
    
    # ----------------------------------------------------
    # 2. Auto-crop the image using the largest contour
    # ----------------------------------------------------
    # Convert to grayscale for contour detection
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    
    # Use a blur to reduce noise before thresholding
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
    
    # Binary threshold (OTSU adaptive could also be used)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("[Warning] No contours found. Skipping auto-crop step.")
        cropped_img = original_img.copy()
    else:
        # Sort contours by area; largest should be our object of interest
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Crop the image to the bounding rectangle
        cropped_img = original_img[y : y + h, x : x + w]
        print(f"[Info] Cropped image to largest contour. Bounding box: (x={x}, y={y}, w={w}, h={h})")
    
    # Save intermediate result for debugging
    cv2.imwrite("1_cropped_image.jpg", cropped_img)
    print("[Info] Cropped image saved as 1_cropped_image.jpg")
    
    # ----------------------------------------------------
    # 3. Enhance the cropped image
    # ----------------------------------------------------
    # We'll do a brightness/contrast adjustment here. 
    # For brightness/contrast, we use the form:
    #    new_image = alpha * image + beta
    # where alpha (1.0-3.0) is contrast, beta (0-100) is brightness.
    # Feel free to tune alpha and beta or make them command-line arguments.
    
    alpha = 1.2  # contrast control
    beta = 20    # brightness control
    
    enhanced_img = cv2.convertScaleAbs(cropped_img, alpha=alpha, beta=beta)
    
    # Optional: Sharpen the image with a kernel
    # Example sharpening kernel
    sharpen_kernel = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ], dtype=np.float32)
    
    enhanced_img = cv2.filter2D(enhanced_img, -1, sharpen_kernel)
    
    # Save intermediate result
    cv2.imwrite("2_enhanced_image.jpg", enhanced_img)
    print("[Info] Enhanced image saved as 2_enhanced_image.jpg")
    
    # ----------------------------------------------------
    # 4. Detect edges (demonstration of a detection step)
    # ----------------------------------------------------
    # Canny edge detection is a straightforward approach. 
    # For advanced object detection (like YOLO, SSD, etc.), 
    # you would integrate a pre-trained model here.
    
    # Convert to grayscale (for edge detection)
    gray_enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny
    # The thresholds can be tweaked for sensitivity
    edges = cv2.Canny(gray_enhanced_img, threshold1=50, threshold2=150)
    
    # Optional: If we want to stack edges back in color for debugging
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # Save result
    cv2.imwrite("3_detected_edges.jpg", edges_bgr)
    print("[Info] Edge detection output saved as 3_detected_edges.jpg")
    
    # ----------------------------------------------------
    # 5. Finish up
    # ----------------------------------------------------
    print("[Info] Processing completed successfully.")

if __name__ == "__main__":
    main()