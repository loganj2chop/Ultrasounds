import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm

print("Starting underlay generation...")

# =========================================================
# Load Ultrasound Images Only
# =========================================================
image_array = np.load('ultrasound_RPA_images_256.npy')
print("Original image array shape:", image_array.shape)

images = torch.tensor(image_array, dtype=torch.float32)

# Add channel dimension if needed
if images.ndim == 3:
    images = images.unsqueeze(1)

print("Tensor shape:", images.shape)

# =========================================================
# Dataset (no masks needed)
# =========================================================
class KidneyImageDataset(Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]

dataset = KidneyImageDataset(images)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

# =========================================================
# Load Pretrained Kidney Segmentation Model
# =========================================================
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=1,
    classes=1
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Running on device:", device)

model.load_state_dict(torch.load('model_pretrained.pth', map_location=device))
model = model.to(device)
model.eval()

# =========================================================
# Generate Kidney Masks
# =========================================================
all_underlays = []

with torch.no_grad():
    for batch in tqdm(dataloader):
        batch = batch.to(device)

        outputs = model(batch)
        preds = torch.sigmoid(outputs)
        preds = (preds > 0.5).float()

        # Multiply original image by predicted kidney mask
        underlay = batch * preds

        all_underlays.append(underlay.cpu())

# =========================================================
# Combine and Save
# =========================================================
underlays_tensor = torch.cat(all_underlays, dim=0)
underlays_np = underlays_tensor.squeeze(1).numpy()

print("Underlay shape:", underlays_np.shape)

np.save("ultrasound_RPA_underlays_256.npy", underlays_np)

print("Saved ultrasound_underlays_256.npy")