import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Dataset , Sampler
from torch.nn.utils import spectral_norm
from PIL import Image
import numpy as np
from pytorch_msssim import ms_ssim


FOLDER_NAME = "[PLACEHOLDER]" # FOLDER TO SAVE SAMPLE GENERATED IMAGES
GENERATOR_MODEL_NAME = "[PLACEHODLER].pth" # NAME TO SAVE GENERATOR MODEL 
DISCRIMINATOR_MODEL_NAME = "[PLACEHOLDER].pth"# NAME TO SAVE DISCRIMINATIR MODEL

# Hyperparameters
EPOCHS = 301  
BATCH_SIZE = 8
IMAGE_SIZE = 512
LATENT_DIM = 320 
NUM_CLASSES = 6  # Number of classes in dataset
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_EVERY = 10 

os.makedirs(FOLDER_NAME, exist_ok=True)
os.makedirs("[PLACEHOLDER_C]/best_overall", exist_ok=True)  #FOLDER TO SAVE BEST OVERALL MODELS
os.makedirs("[PLACEHOLDER_C]/every_10_epochs", exist_ok=True)  #FOLDER TO SAVE MODELS EVERY 10 EPOCHS
os.makedirs("[PLACEHOLDER_C]/final_model", exist_ok=True) #FOLDER TO SAVE FINAL MODEL(FINAL EPOCH)

for class_id in range(NUM_CLASSES):
    os.makedirs(f"[PLACEHOLDER_C]/best_class_{class_id}", exist_ok=True)

# Dataset Class
class CTScanDataset(Dataset):
    def __init__(self, root_folder, transform=None):
        self.root_folder = root_folder
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {cls: idx for idx, cls in enumerate(os.listdir(root_folder))}
        
        for class_name in os.listdir(root_folder):
            class_folder = os.path.join(root_folder, class_name)
            if os.path.isdir(class_folder):
                for img_name in os.listdir(class_folder):
                    self.image_paths.append(os.path.join(class_folder, img_name))
                    self.labels.append(self.class_to_idx[class_name])

        #Ensure balanced class distribution when shuffling data
        self.data = list(zip(self.image_paths, self.labels))
        np.random.shuffle(self.data)

    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("L") 
        if self.transform:
            image = self.transform(image)
        return image, label

# Data Transform
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Single mean & std for 1 channel
])

class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, num_classes, batch_size):
        self.dataset = dataset
        self.num_classes = num_classes
        self.batch_size = batch_size

        labels_array = np.array(dataset.labels)
        self.class_indices = {c: np.where(labels_array == c)[0].tolist() for c in range(num_classes)}

        # Ensure at least one sample per class
        for c in self.class_indices:
            if len(self.class_indices[c]) == 0:
                raise ValueError(f"Class {c} has no samples!")

        # Shuffle indices once to ensure diversity
        self.class_indices = {c: np.random.permutation(self.class_indices[c]).tolist() for c in range(num_classes)}
        self.current_index = {c: 0 for c in range(num_classes)} 

    def __iter__(self):
        per_class = self.batch_size // self.num_classes
        remainder = self.batch_size % self.num_classes

        indices = []
        for _ in range(len(self.dataset) // self.batch_size):
            batch = []
            for c in range(self.num_classes):
                start = self.current_index[c]
                end = start + per_class
                batch.extend(self.class_indices[c][start:end])

                # Update index and reshuffle if needed
                self.current_index[c] = end
                if self.current_index[c] >= len(self.class_indices[c]):
                    self.current_index[c] = 0
                    np.random.shuffle(self.class_indices[c])  # Reshuffle to get new samples

            # Add extra random samples if batch_size isn't perfectly divisible
            extra_classes = np.random.choice(self.num_classes, remainder, replace=True)
            for c in extra_classes:
                batch.append(self.class_indices[c][self.current_index[c]])
                self.current_index[c] += 1
                if self.current_index[c] >= len(self.class_indices[c]):
                    self.current_index[c] = 0
                    np.random.shuffle(self.class_indices[c])

            np.random.shuffle(batch)  
            indices.extend(batch)
        return iter(indices)

    def __len__(self):
        return len(self.dataset) // self.batch_size

if __name__ == '__main__':

    dataset = CTScanDataset("[DATASET]", transform=transform) #FOLDER WIHT DATASET
    balanced_sampler = BalancedBatchSampler(dataset, NUM_CLASSES, BATCH_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=balanced_sampler, num_workers=4)


    # CGAN Model
    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()
            self.label_embedding = nn.Embedding(NUM_CLASSES, LATENT_DIM * 4)
            self.fc = nn.Linear(LATENT_DIM * 5, 512 * 4 * 4)

            self.model = nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.Unflatten(1, (512, 4, 4)), 

                #(4x4 → 8x8)
                nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2),

                nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),  # 8x8 → 16x16
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),

                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 16x16 → 32x32
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),

                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 32x32 → 64x64
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2),

                nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 64x64 → 128x128
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.2),

                nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),  # 128x128 → 256x256
                nn.BatchNorm2d(16),
                nn.LeakyReLU(0.2),

                nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),  # 256x256 → 512x512
                nn.Tanh()
            )

        def forward(self, noise, labels):
            label_embedding = self.label_embedding(labels)
            gen_input = torch.cat((noise, label_embedding), -1)
            gen_input = self.fc(gen_input)
            return self.model(gen_input)

    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            self.label_embedding = nn.Embedding(NUM_CLASSES, 1)

            self.model = nn.Sequential(
                spectral_norm(nn.Conv2d(2, 64, 4, stride=2, padding=1)),  # 512x512 → 256x256
                nn.LeakyReLU(0.2),

                spectral_norm(nn.Conv2d(64, 128, 4, stride=2, padding=1)),  # 256x256 → 128x128
                nn.LeakyReLU(0.2),

                spectral_norm(nn.Conv2d(128, 256, 4, stride=2, padding=1)),  # 128x128 → 64x64
                nn.LeakyReLU(0.2),

                spectral_norm(nn.Conv2d(256, 512, 4, stride=2, padding=1)),  # 64x64 → 32x32
                nn.LeakyReLU(0.2),

                spectral_norm(nn.Conv2d(512, 1024, 4, stride=2, padding=1)),  # 32x32 → 16x16
                nn.LeakyReLU(0.2),

                nn.Conv2d(1024, 1024, 4, stride=2, padding=1),  # 16x16 → 8x8
                nn.LeakyReLU(0.2),

                nn.Conv2d(1024, 1, 4, stride=1),  # 8x8 → 1x1
                nn.AdaptiveAvgPool2d(1)  # Output single value
            )

        def forward(self, img, labels):
            label_embedding = self.label_embedding(labels).unsqueeze(2).unsqueeze(3)
            label_embedding = label_embedding.expand(-1, 1, img.shape[2], img.shape[3]) 

            d_input = torch.cat((img, label_embedding), dim=1) 
            out = self.model(d_input)
            return out.view(out.shape[0], 1)  

    # Initialize Models
    generator = Generator().to(DEVICE)
    discriminator = Discriminator().to(DEVICE)

    # Loss & Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0004, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, mode='min', factor=0.7, patience=10, min_lr=1e-6)
    scheduler_D = optim.lr_scheduler.ReduceLROnPlateau(optimizer_D, mode='min', factor=0.7, patience=10, min_lr=1e-6)

    best_msssim = 0.0
    if os.path.exists(GENERATOR_MODEL_NAME) and os.path.exists(DISCRIMINATOR_MODEL_NAME):
        generator.load_state_dict(torch.load(GENERATOR_MODEL_NAME))  
        discriminator.load_state_dict(torch.load(DISCRIMINATOR_MODEL_NAME))  
        print("Loaded previously saved best model to continue training.")
    else:
        print("No saved model found, starting fresh.")

    def balanced_sampling(batch_size, num_classes, device):
        """Ensure each class appears at least once in the batch."""
        per_class = batch_size // num_classes  # How many samples per class
        remainder = batch_size % num_classes  # Leftover samples

        labels = torch.cat([
            torch.full((per_class,), class_idx, dtype=torch.long, device=device)
            for class_idx in range(num_classes)
        ])

        # Add remainder classes randomly
        if remainder > 0:
            extra_labels = torch.randint(0, num_classes, (remainder,), device=device)
            labels = torch.cat([labels, extra_labels])

        return labels[torch.randperm(batch_size)]  # Shuffle

    best_msssim_per_class = {class_id: 0.0 for class_id in range(NUM_CLASSES)}

    for epoch in range(EPOCHS):  
        for i, (imgs, labels) in enumerate(dataloader):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            batch_size = imgs.shape[0]

            noise = torch.randn(batch_size, LATENT_DIM).to(DEVICE)
            sampled_labels = balanced_sampling(batch_size, NUM_CLASSES, DEVICE)
            optimizer_D.zero_grad()
            optimizer_G.zero_grad()

            for _ in range(1):  # Discriminator Update
                with torch.no_grad():
                    fake_imgs = generator(noise, sampled_labels).detach()  

                real_preds = discriminator(imgs, labels)  # Real images with correct labels
                fake_preds = discriminator(fake_imgs, sampled_labels)  # Fake images with sampled labels

                real_loss = criterion(real_preds, torch.ones_like(real_preds))  # Labels = 1
                fake_loss = criterion(fake_preds, torch.zeros_like(fake_preds))  # Labels = 0
                d_loss = real_loss + fake_loss 

                d_loss.backward()
                optimizer_D.step()

            # Train Generator
            for _ in range(2): # Generator Update
                fake_imgs = generator(noise, sampled_labels)

                fake_preds = discriminator(fake_imgs, sampled_labels)  # Discriminator output for fake images
                g_loss = criterion(fake_preds, torch.ones_like(fake_preds))  

                g_loss.backward()
                optimizer_G.step()

        print(f"Epoch {epoch}/{EPOCHS-1} - D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

        if epoch > 15:
            scheduler_G.step(g_loss)  # Adjust G LR based on G loss
            scheduler_D.step(d_loss)  # Adjust D LR based on D loss

        # Print current learning rates
        current_lr_G = optimizer_G.param_groups[0]['lr']
        current_lr_D = optimizer_D.param_groups[0]['lr']
        print(f"Epoch {epoch}: G LR = {current_lr_G:.6f}, D LR = {current_lr_D:.6f}")

        # Compute MS-SSIM 
        with torch.no_grad():
            noise = torch.randn(batch_size, LATENT_DIM).to(DEVICE)
            sample_labels = balanced_sampling(batch_size, NUM_CLASSES, DEVICE)
            fake_imgs = generator(noise, sample_labels)

        # Store MS-SSIM scores for each class
            msssim_scores_per_class = {}

            for class_label in range(NUM_CLASSES):
            # Get indices of images for the current class
                real_class_indices = (labels == class_label).nonzero(as_tuple=True)[0]
                fake_class_indices = (sample_labels == class_label).nonzero(as_tuple=True)[0]

                if len(real_class_indices) > 0 and len(fake_class_indices) > 0:
                    #real_class_imgs = imgs[real_class_idx]
                    #fake_class_imgs = fake_imgs[fake_class_idx]
                    real_class_imgs = imgs[real_class_indices]
                    fake_class_imgs = fake_imgs[fake_class_indices]

                    min_samples = min(len(real_class_imgs), len(fake_class_imgs))
                    real_class_imgs = real_class_imgs[:min_samples]
                    fake_class_imgs = fake_class_imgs[:min_samples]

                    msssim_score = ms_ssim(fake_class_imgs, real_class_imgs, data_range=1.0, size_average=True).item()
                    msssim_scores_per_class[class_label] = msssim_score
                else:
                    msssim_scores_per_class[class_label] = None  # No images for this class

        # Compute the overall average MS-SSIM score across all classes (excluding None values)
            valid_scores = [score for score in msssim_scores_per_class.values() if score is not None]
            average_msssim = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

            print(f"Epoch {epoch}: MS-SSIM Scores per Class = {msssim_scores_per_class}")
            print(f"Epoch {epoch}: Average MS-SSIM Score = {average_msssim:.4f}")

                # SAVE BEST MODELS PER CLASS
            for class_label, score in msssim_scores_per_class.items():
                if score is not None and score > best_msssim_per_class[class_label]:
                    best_msssim_per_class[class_label] = score
                    torch.save(generator.state_dict(), f"[PLACEHOLDER_C]/best_class_{class_label}/generator_epoch_{epoch}.pth")
                    torch.save(discriminator.state_dict(), f"[PLACEHOLDER_C]/best_class_{class_label}/discriminator_epoch_{epoch}.pth")
                    print(f"Saved new best model for Class {class_label} at epoch {epoch} with MS-SSIM {score:.4f}")

            # SAVE THE BEST OVERALL MODEL BASED ON AVERAGE MS-SSIM
            if average_msssim > best_msssim:
                best_msssim = average_msssim
                torch.save(generator.state_dict(), "[PLACEHOLDER_C]/best_overall/generator_best.pth")
                torch.save(discriminator.state_dict(), "[PLACEHOLDER_C]/best_overall/discriminator_best.pth")
                print(f"Saved new best overall model at epoch {epoch} with MS-SSIM {average_msssim:.4f}")

            #  SAVE MODELS EVERY 10 EPOCHS
            if epoch % 10 == 0:
                torch.save(generator.state_dict(), f"[PLACEHOLDER_C]/every_10_epochs/generator_epoch_{epoch}.pth")
                torch.save(discriminator.state_dict(), f"[PLACEHOLDER_C]/every_10_epochs/discriminator_epoch_{epoch}.pth")
                print(f"Saved model at epoch {epoch} (every 10 epochs).")

        
        # Save Sample Images 
        if epoch % SAVE_EVERY == 0:
            with torch.no_grad():
                sample_noise = torch.randn(NUM_CLASSES, LATENT_DIM).to(DEVICE)
                sample_labels = torch.arange(NUM_CLASSES).to(DEVICE)
                samples = generator(sample_noise, sample_labels)

                for i, sample in enumerate(samples):
                    vutils.save_image(sample,f"{FOLDER_NAME}/epoch_{epoch}_class_{i}.png", normalize=True)

    # SAVE FINAL MODEL AT END OF TRAINING
    torch.save(generator.state_dict(), "[PLACEHOLDER_C]/final_model/generator_final.pth")
    torch.save(discriminator.state_dict(), "[PLACEHOLDER_C]/final_model/discriminator_final.pth")
    print(f"Saved final model at epoch {EPOCHS - 1}.")