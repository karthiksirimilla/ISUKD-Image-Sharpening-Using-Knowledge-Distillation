import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from models.student_model import VideoSharpeningStudent
import numpy as np

# ---- Dataset ----
class DIV2KPairedDataset(Dataset):
    def __init__(self, sharp_dir, img_size=256):
        self.sharp_paths = sorted([os.path.join(sharp_dir, f) for f in os.listdir(sharp_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
    def __len__(self):
        return len(self.sharp_paths)
    def __getitem__(self, idx):
        sharp = Image.open(self.sharp_paths[idx]).convert('RGB')
        sharp_np = np.array(sharp)
        # Simulate blur (downscale + upscale)
        blurry_np = np.array(sharp.resize((sharp.width//4, sharp.height//4), Image.BICUBIC).resize(sharp.size, Image.BICUBIC))
        blurry = Image.fromarray(blurry_np)
        return self.transform(blurry), self.transform(sharp)

# ---- Model Setup ----
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print("Using device:", device)

# Load pretrained teacher (Restormer)
teacher = torch.hub.load('swz30/Restormer', 'Restormer').eval().to(device)
for p in teacher.parameters():
    p.requires_grad = False

# Load student (your lightweight model)
student = VideoSharpeningStudent().to(device)

# ---- Training Setup ----
train_dataset = DIV2KPairedDataset('DIV2K/DIV2K_train_HR', img_size=256)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)

optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)
l1_loss = nn.L1Loss()
mse_loss = nn.MSELoss()

# ---- Training Loop ----
epochs = 10  # Adjust as needed
for epoch in range(epochs):
    student.train()
    running_loss = 0.0
    for blurry, sharp in train_loader:
        blurry, sharp = blurry.to(device), sharp.to(device)
        with torch.no_grad():
            teacher_out = teacher(blurry)
        student_out = student(blurry)
        loss = (
            1.0 * l1_loss(student_out, sharp) +
            0.1 * mse_loss(student_out, teacher_out)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f}")

# ---- Save Weights ----
os.makedirs('models', exist_ok=True)
torch.save(student.state_dict(), 'models/student_model.pth')
print("Training complete. Student weights saved to models/student_model.pth") 