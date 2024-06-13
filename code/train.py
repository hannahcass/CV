import numpy as np
import torch
from architectures import NewNorm2b
from datasets import DatasetRandomCrop
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from utilities import train_auto_encoder

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

### Path to folder with imput images
root_dir=r"D:\JKU\JKU\!Computer_Vision\project\integrated_images"

### Hyperparameters
epochs = 6000
batch_size = 64
lr = 2e-4
adjust_lr = []
adjust_amount = 0.75

dataset = DatasetRandomCrop(root_dir, size=128)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = NewNorm2b().to(device)
loss_func = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

loss = train_auto_encoder(
    auto_encoder=model, loader=train_loader, objective=loss_func, optimiser=optimizer, num_epochs=epochs, adjust_lr=adjust_lr, adjust_amount=adjust_amount)