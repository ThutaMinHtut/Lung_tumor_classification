import os
import cv2
import numpy as np
from PIL import Image

# Paths
input_folder = "ResizedDataset/train"
output_folder = "Masks/train"

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# Loop through all categories
for category in os.listdir(input_folder):
    category_path = os.path.join(input_folder, category)
    mask_category_path = os.path.join(output_folder, category)

    os.makedirs(mask_category_path, exist_ok=True)

    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)
        save_mask_path = os.path.join(mask_category_path, img_name)

        # Read image
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Apply thresholding to detect tumors
        _, mask = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

        # Save mask
        cv2.imwrite(save_mask_path, mask)

print("Masks generated and saved in:", output_folder)
