import numpy as np
from PIL import Image
import os

input_dir = "Lung_tumor_detection/LungcancerDataSet/Data"
output_dir = "Lung_tumor_detection/ResizedDataset"

for subdir in ['train', 'test', 'valid']:
    for category in os.listdir(f"{input_dir}/{subdir}"):
        input_path = f"{input_dir}/{subdir}/{category}"
        output_path = f"{output_dir}/{subdir}/{category}"
        os.makedirs(output_path, exist_ok=True)

        for file in os.listdir(input_path):
            img = Image.open(os.path.join(input_path, file))
            img = img.resize((224, 224))
            img.save(os.path.join(output_path, file))

