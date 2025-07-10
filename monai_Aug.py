import os
import numpy as np
import torch
import imageio.v2 as imageio
from monai.transforms import (
    Compose, LoadImage, Resize, RandAffine, RandFlip, RandAdjustContrast,
    Rand2DElastic, RandGaussianNoise, RandGaussianSmooth, RandHistogramShift, RandLambda,RandRotate, RandZoom,RandGaussianSharpen
)

# Define paths
root_dir = "[PLACEHOLDER_A]" #FOLDER WITH IMAGES TO BE AUGMENTED
output_root = "[PLACEHOLDER_B]" #FOLDER TO SAVE NEW AUGMENTED IMAGES
splits = ["train", "valid", "test"]
classes = ["adenocarcinoma", "benign", "large.cell.carcinoma", "malignant", "normal", "squamous.cell.carcinoma"]

# Define the target number of images per class (for balancing)
target_num = 500  

augmentations = Compose([
    RandFlip(prob=0.5, spatial_axis=1),  # 50% Horizontal Flip
    RandAffine(prob=0.8, rotate_range=(-0.26, 0.26), translate_range=(50, 50), scale_range=(-0.2, 0.2), padding_mode="border"),  # **Rotation (-15° to 15°), Shift (10%), Zoom (0.8x – 1.2x)**
    #Rand2DElastic(prob=0.3, spacing=(20, 20), magnitude_range=(1, 2)),
    #RandGaussianNoise(prob=0.2, mean=0, std=0.005),
    #RandGaussianSmooth(prob=0.3, sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5)),
    #RandHistogramShift(prob=0.2, num_control_points=5),
    RandAdjustContrast(prob=0.3, gamma=(0.9, 1.1)),
    #RandLambda(prob=0.3, func=lambda x: np.power(x, np.random.uniform(0.9, 1.1)))  # **Gamma correction**
])


# applied after augmentation
resize_transform = Resize((224, 224))

# Process dataset
for split in splits:
    input_split_dir = os.path.join(root_dir, split)
    output_split_dir = os.path.join(output_root, split)

    if not os.path.exists(output_split_dir):
        os.makedirs(output_split_dir)

    for cls in classes:
        input_class_dir = os.path.join(input_split_dir, cls)
        output_class_dir = os.path.join(output_split_dir, cls)
        if not os.path.exists(output_class_dir):
            os.makedirs(output_class_dir)

        image_files = [f for f in os.listdir(input_class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        print(f"Processing split '{split}', class '{cls}' with {len(image_files)} original images.")

        processed_images = []
        
        # Resize all images
        for img_file in image_files:
            img_path = os.path.join(input_class_dir, img_file)

            
            image = LoadImage(image_only=True, reader="PILReader", reverse_indexing=False)(img_path).astype(np.float32) / 255.0  # Normalize

            #Convert RGBA/RGB to grayscale
            if image.ndim == 3 and image.shape[-1] in [3, 4]:  
                image = np.mean(image[:, :, :3], axis=-1)  # Convert RGB(A) to grayscale

            # **Ensure shape is (1, H, W)**
            image = np.expand_dims(image, axis=0)

            
            #resized_img = resize_transform(image)

            # Convert back to uint8 
            resized_img = image.squeeze(0) 
            resized_img = (resized_img * 255).astype(np.uint8)

            # **Save resized images**
            out_filename = os.path.join(output_class_dir, f"{os.path.splitext(img_file)[0]}_resized.png")
            imageio.imwrite(out_filename, resized_img)

            processed_images.append(image)  # Store for aug

        
        if split == "train":
            current_count = len(processed_images)
            num_to_generate = target_num - current_count

            if num_to_generate > 0:
                print(f"Augmenting class '{cls}' in train split: Generating {num_to_generate} images.")

                augmented_count = 0
                while augmented_count < num_to_generate:
                    for img in processed_images:
                        if augmented_count >= num_to_generate:
                            break 

                        
                        aug_img = augmentations(img)

                        # Resize after augmentation to 224x224
                        #aug_img = resize_transform(aug_img)

                        # Convert back to uint8 for saving
                        aug_img = aug_img.squeeze(0)  
                        aug_img = (aug_img * 255).astype(np.uint8)

                        out_filename = os.path.join(output_class_dir, f"{cls}_aug_{augmented_count}.png")
                        imageio.imwrite(out_filename, aug_img)
                        augmented_count += 1

            print(f"Finished balancing class '{cls}' in train split to {target_num} images.")

print("Augmented images saved in:", output_root)

