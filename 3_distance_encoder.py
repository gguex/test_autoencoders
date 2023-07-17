import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Dataset

mnist_dataset = MNIST("data/", train=True, download=True, transform=transforms.ToTensor())
mnist_dataloader = DataLoader(mnist_dataset, batch_size=64, shuffle=True)

# Model definition

class DistEncoder(nn.Module):

  def __init__(self, input_dim, low_dim):
    
    super(DistEncoder, self).__init__()
    
    self.encoder = torch.nn.Sequential(
      nn.Linear(input_dim, 256),
      nn.ReLU(),
      nn.Linear(256, 64),
      nn.ReLU(),
      nn.Linear(64, 16),
      nn.ReLU(),
      nn.Linear(16, 8),
      nn.ReLU(),
      nn.Linear(8, low_dim)
      )
      
  def forward(self, input_vec_1, input_vec_2):
    low_vec_1 = self.encoder(input_vec_1)
    low_vec_2 = self.encoder(input_vec_2)
    dist = torch.sum((low_vec_1 - low_vec_2)**2, 1)
    return dist

de_model = DistEncoder(784, 2).to(device)
print(de_model)

# Training 

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(de_model.parameters(), lr=0.01)

n_epochs = 10
de_model = de_model.train()

for epoch in range(n_epochs):

    train_loss = 0

    for inputs, _ in mnist_dataloader:

        inputs = inputs.flatten(1).to(device)
        sample_size = inputs.shape[0]
        
        inputs_1 = inputs[:int(sample_size/2), :]
        inputs_2 = inputs[int(sample_size/2):, :]
        
        optimizer.zero_grad()

        estimated_d = de_model(inputs_1, inputs_2)
        
        loss = loss_fn(estimated_d, torch.sum((inputs_1 - inputs_2)**2, 1))
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print(f"Epoch {epoch} : "
          f"train loss = {train_loss / len(mnist_dataloader.dataset)} ")


# Transformation of data

de_model = de_model.eval()
low_dim_vectors = np.empty((0, 2))
output_v = np.empty((0))
for inputs, outputs in mnist_dataloader:

  inputs = inputs.flatten(1).to(device)

  low_coord = de_model.encoder(inputs)
  low_dim_vectors = np.concatenate((low_dim_vectors,
                                    np.array(low_coord.cpu().detach().numpy())))
  output_v = np.concatenate((output_v, outputs))

# Plot 

sns.set(context="paper", style="white")
fig, ax = plt.subplots(figsize=(12, 10))
color = output_v
plt.scatter(low_dim_vectors[:, 0], low_dim_vectors[:, 1], c=color,
            cmap="Spectral", s=0.1)
plt.setp(ax, xticks=[], yticks=[])
plt.title("MNIST data embedded into two dimensions by Distance Encoder",
          fontsize=18)
plt.show()