"""
MNIST Digit Generation using GAN
This script trains a Generative Adversarial Network (GAN) to generate handwritten digits
similar to those in the MNIST dataset.
"""

import math
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ---------------------------
# Data Loading and Preparation
# ---------------------------

def load_and_explore_data():
    """Load MNIST dataset and display basic statistics"""
    # Define image transformation (convert to tensor)
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Load training dataset
    train_ds = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    # Print dataset statistics
    print("Dataset shape:", train_ds.data.shape)
    print("Targets shape:", train_ds.targets.shape)
    print("Classes:", train_ds.classes)
    
    # Analyze first image
    first_image = train_ds.data[0]
    print("\nFirst image values:\n", first_image)
    print("First image label:", train_ds.targets[0].item())
    print("Max pixel value:", first_image.max().item())
    print("Min pixel value:", first_image.min().item())
    print("Mean pixel value:", first_image.float().mean().item())
    print("Pixel std deviation:", first_image.float().std().item())
    
    return train_ds

# ---------------------------
# Neural Network Definitions
# ---------------------------

class Discriminator(nn.Module):
    """Discriminator network to differentiate real and fake images"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Network architecture
        self.fc1 = nn.Linear(in_features, 128)
        self.leaky_relu1 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(128, 64)
        self.leaky_relu2 = nn.LeakyReLU(0.2)
        self.fc3 = nn.Linear(64, 32)
        self.leaky_relu3 = nn.LeakyReLU(0.2)
        self.fc4 = nn.Linear(32, out_features)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Flatten input image
        x = x.view(x.size(0), -1)
        
        # Forward pass through layers
        x = self.leaky_relu1(self.fc1(x))
        x = self.dropout(x)
        x = self.leaky_relu2(self.fc2(x))
        x = self.dropout(x)
        x = self.leaky_relu3(self.fc3(x))
        x = self.dropout(x)
        return self.fc4(x)

class Generator(nn.Module):
    """Generator network to create fake images"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Network architecture
        self.fc1 = nn.Linear(in_features, 32)
        self.relu1 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(32, 64)
        self.relu2 = nn.LeakyReLU(0.2)
        self.fc3 = nn.Linear(64, 128)
        self.relu3 = nn.LeakyReLU(0.2)
        self.fc4 = nn.Linear(128, out_features)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Forward pass through layers
        x = self.relu1(self.fc1(x))
        x = self.dropout(x)
        x = self.relu2(self.fc2(x))
        x = self.dropout(x)
        x = self.relu3(self.fc3(x))
        x = self.dropout(x)
        return self.tanh(self.fc4(x))

# ---------------------------
# Training Utilities
# ---------------------------

def real_loss(predicted, loss_fn, device):
    """Calculate loss for real samples (target = 1)"""
    batch_size = predicted.size(0)
    targets = torch.ones(batch_size).to(device)
    return loss_fn(predicted.squeeze(), targets)

def fake_loss(predicted, loss_fn, device):
    """Calculate loss for fake samples (target = 0)"""
    batch_size = predicted.size(0)
    targets = torch.zeros(batch_size).to(device)
    return loss_fn(predicted.squeeze(), targets)

def display_images(images, n_cols=4, figsize=(12, 6)):
    """Display images in a grid format"""
    plt.style.use('ggplot')
    n_images = len(images)
    n_rows = math.ceil(n_images / n_cols)
    
    plt.figure(figsize=figsize)
    for idx in range(n_images):
        ax = plt.subplot(n_rows, n_cols, idx+1)
        img = images[idx].permute(1, 2, 0)  # CHW to HWC
        cmap = 'gray' if img.shape[-1] == 1 else None
        ax.imshow(img.squeeze(), cmap=cmap)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# ---------------------------
# Training Loop
# ---------------------------

def train_gan(d, g, d_optim, g_optim, loss_fn, loader, epochs, device):
    """Main training loop for GAN"""
    print(f"Training on {device}...")
    
    # Initialize tracking variables
    fixed_z = torch.randn(16, 100).to(device)  # For visualization
    fixed_samples = []
    d_losses = []
    g_losses = []
    
    # Move models to device
    d.to(device)
    g.to(device)
    
    for epoch in range(epochs):
        d.train()
        g.train()
        d_running_loss = 0.0
        g_running_loss = 0.0
        
        for real_images, _ in loader:
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            
            # --- Train Discriminator ---
            d_optim.zero_grad()
            
            # Real images loss
            real_images = real_images * 2 - 1  # Scale to [-1, 1]
            d_real = d(real_images)
            d_loss_real = real_loss(d_real, loss_fn, device)
            
            # Fake images loss
            z = torch.randn(batch_size, 100).to(device)
            with torch.no_grad():
                fake_images = g(z)
            d_fake = d(fake_images)
            d_loss_fake = fake_loss(d_fake, loss_fn, device)
            
            # Total discriminator loss
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optim.step()
            
            # --- Train Generator ---
            g_optim.zero_grad()
            
            z = torch.randn(batch_size, 100).to(device)
            gen_images = g(z)
            d_output = d(gen_images)
            g_loss = real_loss(d_output, loss_fn, device)
            
            g_loss.backward()
            g_optim.step()
            
            # Track losses
            d_running_loss += d_loss.item()
            g_running_loss += g_loss.item()
        
        # Save epoch losses
        d_epoch_loss = d_running_loss / len(loader)
        g_epoch_loss = g_running_loss / len(loader)
        d_losses.append(d_epoch_loss)
        g_losses.append(g_epoch_loss)
        
        # Generate sample images
        g.eval()
        with torch.no_grad():
            fixed_samples.append(g(fixed_z).cpu())
        
        print(f"Epoch [{epoch+1}/{epochs}] | D Loss: {d_epoch_loss:.4f} | G Loss: {g_epoch_loss:.4f}")
    
    # Save generated samples
    with open('fixed_samples.pkl', 'wb') as f:
        pkl.dump(fixed_samples, f)
    
    return d_losses, g_losses

# ---------------------------
# Main Execution
# ---------------------------

if __name__ == "__main__":
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    latent_size = 100
    image_size = 28 * 28  # MNIST dimensions
    epochs = 10
    batch_size = 2024
    
    # Load data
    train_ds = load_and_explore_data()
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    # Initialize models
    discriminator = Discriminator(image_size, 1)
    generator = Generator(latent_size, image_size)
    
    # Optimizers
    d_optim = optim.Adam(discriminator.parameters(), lr=0.002)
    g_optim = optim.Adam(generator.parameters(), lr=0.002)
    
    # Loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # Start training
    d_losses, g_losses = train_gan(
        discriminator,
        generator,
        d_optim,
        g_optim,
        criterion,
        loader,
        epochs,
        device
    )
    
    # Plot training progress
    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label='Discriminator')
    plt.plot(g_losses, label='Generator')
    plt.title("Training Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    
    # Generate final sample
    generator.eval()
    with torch.no_grad():
        z = torch.randn(1, 100).to(device)
        sample = generator(z).cpu()
        sample = (sample + 1) / 2  # Scale from [-1,1] to [0,1]
        display_images(sample.view(1, 1, 28, 28), n_cols=1, figsize=(2, 2))