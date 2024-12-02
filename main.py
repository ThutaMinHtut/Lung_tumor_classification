import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from torchvision.models import ResNet50_Weights


# Calculate class weights
from sklearn.utils.class_weight import compute_class_weight
import numpy as np



# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ResNet-50 normalization
])

# Directories
train_dir = "Lung_tumor_detection/ResizedDataset/train"
valid_dir = "Lung_tumor_detection/ResizedDataset/valid"
test_dir = "Lung_tumor_detection/ResizedDataset/test"

# Datasets and loaders
train_data = datasets.ImageFolder(train_dir, transform=transform)
valid_data = datasets.ImageFolder(valid_dir, transform=transform)
test_data = datasets.ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)


####################################################################################


# Get class counts and compute weights
class_counts = [len([f for f in train_data.samples if f[1] == idx]) for idx in range(len(train_data.classes))]
class_weights = compute_class_weight(class_weight='balanced', classes=np.array(range(len(train_data.classes))), y=[s[1] for s in train_data.samples])
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

scaled_weights = class_weights * 0.5  # Scale down weights to avoid overcorrection

# Modify the loss function to include class weights
#criterion = nn.CrossEntropyLoss(weight=class_weights)

criterion = nn.CrossEntropyLoss(weight=scaled_weights)

####################################################################################



# Load Pretrained ResNet-50
#model = models.resnet50(pretrained=True)

model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

# Modify the final layer for multi-class classification
num_classes = len(train_data.classes)  # Automatically detects the number of classes
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, num_classes)
)


model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
#optimizer = optim.Adam(model.parameters(), lr=0.001)

optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Reduce learning rate

####################################################################################

def train_model(model, train_loader, valid_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        # Training phase
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation phase
        model.eval()
        valid_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        print(f"Epoch {epoch+1}/{epochs}, "
              f"Train Loss: {running_loss/len(train_loader):.4f}, "
              f"Valid Loss: {valid_loss/len(valid_loader):.4f}, "
              f"Valid Accuracy: {100 * correct/total:.2f}%")

# Train the model
train_model(model, train_loader, valid_loader, criterion, optimizer, epochs=10)

#################################################################################################
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {100 * correct / total:.2f}%")