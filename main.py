import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from torchvision.models import ResNet50_Weights


#for learning rate decay
from torch.optim.lr_scheduler import StepLR

# Calculate class weights
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


#classification report
from sklearn.metrics import classification_report

#oversample
from torch.utils.data import WeightedRandomSampler


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

# Datasets
train_data = datasets.ImageFolder(train_dir, transform=transform)
valid_data = datasets.ImageFolder(valid_dir, transform=transform)
test_data = datasets.ImageFolder(test_dir, transform=transform)


####################################################################################

# Get class counts and compute weights
class_counts = [len([f for f in train_data.samples if f[1] == idx]) for idx in range(len(train_data.classes))]
# Compute class weights
class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
# Assign weights to each sample
sample_weights = [class_weights[label] for _, label in train_data.samples]
# Create the sampler
sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_data.samples), replacement=True)

#class_weights = compute_class_weight(class_weight='balanced', classes=np.array(range(len(train_data.classes))), y=[s[1] for s in train_data.samples])
#class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
#scaled_weights = class_weights * 0.5  # Scale down weights to avoid overcorrection
#criterion = nn.CrossEntropyLoss(weight=scaled_weights)

####################################################################################

# DataLoaders
train_loader = DataLoader(train_data, batch_size=32, sampler=sampler) 
valid_loader = DataLoader(valid_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

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
class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=True):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: No improvement in validation loss for {self.counter} epochs.")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

####################################################################################
scheduler = StepLR(optimizer, step_size=3, gamma=0.1) # decay LR every 3 epoch

def train_model(model, train_loader, valid_loader, criterion, optimizer,scheduler, epochs,save_path):
    early_stopping = EarlyStopping(patience=3)
    best_valid_loss = float('inf')
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

        scheduler.step()

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
        

        best_epoch =None
        # Save the best model based on validation loss
        if valid_loss / len(valid_loader) < best_valid_loss:
            best_valid_loss = valid_loss / len(valid_loader)
            best_epoch= epoch +1
            torch.save(model.state_dict(), save_path)  # Save the model
            print(f"Model saved at epoch {best_epoch} with validation loss: {best_valid_loss:.4f}")
        
        # Check early stopping
        early_stopping(valid_loss / len(valid_loader))
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break

save_path = "best_model.pth"
# Train the model
train_model(model, train_loader, valid_loader, criterion, optimizer,scheduler, epochs=10,save_path=save_path)

#################################################################################################
model.load_state_dict(torch.load("best_model.pth", weights_only=True))
model.eval()
correct = 0
total = 0

# True and predicted labels for classification report
true_labels = []
predicted_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Collect true and predicted labels
        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(preds.cpu().numpy())

print(f"Test Accuracy: {100 * correct / total:.2f}%")

# Generate and print classification report
class_names = train_data.classes  # Class names from training dataset
report = classification_report(true_labels, predicted_labels, target_names=class_names)
print("\nClassification Report:\n")
print(report)

