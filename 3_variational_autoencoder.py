import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import umap
import matplotlib.pyplot as plt
import seaborn as sns

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Dataset

mnist_dataset = MNIST("data/", train=True, download=True, transform=transforms.ToTensor())
mnist_dataloader = DataLoader(mnist_dataset, batch_size=64, shuffle=True)

# Model definition

class VariationalEncoder(nn.Module):
    def __init__(self, input_dim, low_dim):
        super(VariationalEncoder, self).__init__()
        self.pre_encode = torch.nn.Sequential(
          nn.Linear(input_dim, 128),
          nn.ReLU(),
          nn.Linear(128, 36),
          nn.ReLU(),
        )
        self.to_mu = nn.Linear(36, low_dim)
        self.to_sigma = nn.Linear(36, low_dim)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, input_vec):
        pre_vec = self.pre_encode(input_vec)
        mu =  self.to_me(pre_vec)
        sigma = torch.exp(self.to_sigma(pre_vec))
        low_vec = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return low_vec

class Decoder(nn.Module):
    def __init__(self, low_dim, output_dim):
        super(Decoder, self).__init__()
        self.decoder = torch.nn.Sequential(
          nn.Linear(low_dim, 36),
          nn.ReLU(),
          nn.Linear(36, 128),
          nn.ReLU(),
          nn.Linear(128, output_dim)
          nn.Sigmoid()
        )

    def forward(self, input_vec):
        output_vec = self.decoder(input_vec)
        return output_vec




ae_low_model = AELowdim(784, 2).to(device)
print(ae_low_model)


loss_fn = nn.L1Loss()
optimizer = torch.optim.Adam(ae_low_model.parameters(), lr=0.01)

# Training 

n_epochs = 10
ae_low_model = ae_low_model.train()

for epoch in range(n_epochs):

    train_loss = 0

    for inputs, _ in mnist_dataloader:

        inputs = inputs.flatten(1).to(device)

        optimizer.zero_grad()

        estimated_outputs, _ = ae_low_model(inputs)

        loss = loss_fn(estimated_outputs, inputs.flatten(1))
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print(f"Epoch {epoch} : "
          f"train loss = {train_loss / len(mnist_dataloader.dataset)} ")


# Transformation of data

ae_low_model = ae_low_model.eval()
low_dim_vectors = np.empty((0, 2))
output_v = np.empty((0))
for inputs, outputs in mnist_dataloader:

  inputs = inputs.flatten(1).to(device)

  _, low_coord = ae_low_model(inputs)
  low_dim_vectors = np.concatenate((low_dim_vectors,
                                    np.array(low_coord.cpu().detach().numpy())))
  output_v = np.concatenate((output_v, outputs))

sns.set(context="paper", style="white")

fig, ax = plt.subplots(figsize=(12, 10))
color = output_v
plt.scatter(low_dim_vectors[:, 0], low_dim_vectors[:, 1], c=color,
            cmap="Spectral", s=0.1)
plt.setp(ax, xticks=[], yticks=[])
plt.title("MNIST data embedded into two dimensions by AutoEncoder",
          fontsize=18)

plt.show()