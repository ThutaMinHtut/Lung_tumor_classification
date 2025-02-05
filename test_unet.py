import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from train_unet import UNet  # Import U-Net model

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
model.load_state_dict(torch.load("unet_model.pth", map_location=device, weights_only=True))
model.eval()  # Set model to evaluation mode

# Path to test image
test_img_path = "LungcancerDataSet/Data/test/MalignantCases/Malignant case (461).jpg" 
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Load and preprocess the test image
image = Image.open(test_img_path).convert("L")
image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

# Perform segmentation
with torch.no_grad():
    output = model(image_tensor)

# Convert output to binary mask
output_mask = (output.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

# Show original image and segmented mask
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(image, cmap="gray")
ax[0].set_title("Original CT Scan")
ax[1].imshow(output_mask, cmap="jet")
ax[1].set_title("Predicted Tumor Mask")
plt.show()

