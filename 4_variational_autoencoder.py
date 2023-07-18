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

class VariationalEncoder(nn.Module):
    def __init__(self, input_dim, low_dim):
        super(VariationalEncoder, self).__init__()
        self.pre_encode = torch.nn.Sequential(
          nn.Linear(input_dim, 256),
          nn.ReLU(),
          nn.Linear(256, 64),
          nn.ReLU(),
        )
        self.to_mu = nn.Linear(64, low_dim)
        self.to_sigma = nn.Linear(64, low_dim)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, input_vec):
        pre_vec = self.pre_encode(input_vec)
        mu =  self.to_mu(pre_vec)
        sigma = torch.exp(self.to_sigma(pre_vec))
        low_vec = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return low_vec

class Decoder(nn.Module):
    def __init__(self, low_dim, output_dim):
        super(Decoder, self).__init__()
        self.decoder = torch.nn.Sequential(
          nn.Linear(low_dim, 64),
          nn.ReLU(),
          nn.Linear(64, 256),
          nn.ReLU(),
          nn.Linear(256, output_dim),
          nn.Sigmoid()
        )

    def forward(self, input_vec):
        output_vec = self.decoder(input_vec)
        return output_vec

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, low_dim):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(input_dim, low_dim)
        self.decoder = Decoder(low_dim, input_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


var_ae_model = VariationalAutoencoder(784, 2).to(device)
print(var_ae_model)

# Training 

optimizer = torch.optim.Adam(var_ae_model.parameters())

n_epochs = 20
var_ae_model = var_ae_model.train()

for epoch in range(n_epochs):

    train_loss = 0

    for inputs, _ in mnist_dataloader:

        inputs = inputs.flatten(1).to(device)

        optimizer.zero_grad()

        estimated_outputs = var_ae_model(inputs)

        loss = ((estimated_outputs - inputs.flatten(1))**2).sum() + \
            var_ae_model.encoder.kl
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print(f"Epoch {epoch} : "
          f"train loss = {train_loss / len(mnist_dataloader.dataset)} ")


# Transformation of data

var_ae_model = var_ae_model.eval()
low_dim_vectors = np.empty((0, 2))
output_v = np.empty((0))
for inputs, outputs in mnist_dataloader:

  inputs = inputs.flatten(1).to(device)

  low_coord = var_ae_model.encoder(inputs)
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
plt.title("MNIST data embedded into two dimensions by VarAutoEncoder",
          fontsize=18)
plt.show()

# Digit plot 

def plot_reconstructed(autoencoder, r0=(-5, 10), r1=(-10, 5), n=12):
    w = 28
    img = np.zeros((n*w, n*w))
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[x, y]]).to(device)
            x_hat = autoencoder.decoder(z)
            x_hat = x_hat.reshape(28, 28).to('cpu').detach().numpy()
            img[(n-1-i)*w:(n-1-i+1)*w, j*w:(j+1)*w] = x_hat
    plt.imshow(img, extent=[*r0, *r1])

plot_reconstructed(var_ae_model)

# Interpolation

def interpolate(autoencoder, x_1, x_2, n=12):
    z_1 = autoencoder.encoder(x_1.flatten(1))
    z_2 = autoencoder.encoder(x_2.flatten(1))
    z = torch.stack([z_1 + (z_2 - z_1)*t for t in np.linspace(0, 1, n)])
    interpolate_list = autoencoder.decoder(z)
    interpolate_list = interpolate_list.to('cpu').detach().numpy()

    w = 28
    img = np.zeros((w, n*w))
    for i, x_hat in enumerate(interpolate_list):
        img[:, i*w:(i+1)*w] = x_hat.reshape(28, 28)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    
digit_1 = 1
digit_2 = 7

x, y = mnist_dataloader.__iter__().next() # hack to grab a batch
x_1 = x[y == digit_1][1].to(device) # find a 1
x_2 = x[y == digit_2][1].to(device) # find a 0

interpolate(var_ae_model, x_1, x_2, n=30)
