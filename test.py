import os
from PIL import Image
import torch
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to the `test_cases` folder
test_cases_path = "Lung_tumor_detection/Test_cases"

model_path = "best_model.pth" 

# Define class-to-category mapping
cancer_classes = {'Malignant cases', 'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib', 
                  'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa', 
                  'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa'}
no_cancer_classes = {'normal', 'Bengin cases'}


# Load the model
num_classes = 6 
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
model.fc = torch.nn.Sequential(
    torch.nn.Linear(model.fc.in_features, 512),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.4),
    torch.nn.Linear(512, num_classes)
)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()  # Set the model to evaluation mode

# Class names 
class_names = ['Bengin cases', 'Malignant cases', 
               'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib', 
               'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa', 
               'normal', 
               'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa']

# Process each image in the folder
for image_name in os.listdir(test_cases_path):
    image_path = os.path.join(test_cases_path, image_name)

    try:
        image = Image.open(image_path).convert("RGB")  # Load image and convert to RGB
    except Exception as e:
        print(f"Error loading image {image_name}: {e}")
        continue

    input_image = transform(image).unsqueeze(0).to(device)  # Preprocess image

    # Perform inference
    with torch.no_grad():
        output = model(input_image)
        _, predicted_class = torch.max(output, 1)
        class_name = class_names[predicted_class.item()]  # Get class name

    # Determine cancer/no cancer
    if class_name in cancer_classes:
        print(f"Image: {image_name}, Predicted: CANCER ({class_name})")
    elif class_name in no_cancer_classes:
        print(f"Image: {image_name}, Predicted: NO CANCER ({class_name})")
    else:
        print(f"Image: {image_name}, Predicted: UNKNOWN ({class_name})")