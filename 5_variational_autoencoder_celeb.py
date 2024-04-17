import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)



# ------------------------------------------------------------
# --- Constants
# ------------------------------------------------------------

NUM_FEATURES = 128
LOW_DIM = 200
N_EPOCHS = 10
BATCH_SIZE = 128

TRAIN_MODEL = False



# ------------------------------------------------------------
# --- Dataset
# ------------------------------------------------------------

def transform_img(image):
    return transforms.Compose([transforms.Resize((32, 32)),
                               transforms.ToTensor()])(image)

celeb_dataset = ImageFolder(
    "data/celeba-dataset/img_align_celeba", 
    transform=transform_img)
celeb_dataloader = DataLoader(celeb_dataset, batch_size=BATCH_SIZE, 
                              shuffle=True)



# ------------------------------------------------------------
# --- Model definition
# ------------------------------------------------------------

class VariationalEncoder(nn.Module):
    def __init__(self, low_dim):
        super(VariationalEncoder, self).__init__()
        self.pre_encoder = torch.nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=NUM_FEATURES, 
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(NUM_FEATURES),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=NUM_FEATURES, out_channels=NUM_FEATURES, 
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(NUM_FEATURES),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=NUM_FEATURES, out_channels=NUM_FEATURES, 
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(NUM_FEATURES),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=NUM_FEATURES, out_channels=NUM_FEATURES, 
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(NUM_FEATURES),
            nn.LeakyReLU(),
            nn.Flatten()
        )
        self.to_mu = nn.Linear(512, low_dim)
        self.to_log_var = nn.Linear(512, low_dim)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device)
        self.N.scale = self.N.scale.to(device)
        self.kl = 0

    def forward(self, input_vec):
        pre_vec = self.pre_encoder(input_vec)
        mu =  self.to_mu(pre_vec)
        sigma = torch.exp(self.to_log_var(pre_vec) * 0.5).to(device)
        low_vec = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma + mu**2 - torch.log(sigma) - 1/2).sum()
        return low_vec

class Decoder(nn.Module):
    def __init__(self, low_dim):
        super(Decoder, self).__init__()
        self.decoder = torch.nn.Sequential(
          nn.Linear(low_dim, 512),
          nn.BatchNorm1d(512),
          nn.LeakyReLU(),
          nn.Unflatten(1, (NUM_FEATURES, 2, 2)),
          nn.ConvTranspose2d(in_channels=NUM_FEATURES, 
                             out_channels=NUM_FEATURES,
                             kernel_size=3, stride=2, output_padding=1, 
                             padding=1),
          nn.BatchNorm2d(NUM_FEATURES),
          nn.LeakyReLU(),
          nn.ConvTranspose2d(in_channels=NUM_FEATURES, 
                             out_channels=NUM_FEATURES,
                             kernel_size=3, stride=2, output_padding=1, 
                             padding=1),
          nn.BatchNorm2d(NUM_FEATURES),
          nn.LeakyReLU(),
          nn.ConvTranspose2d(in_channels=NUM_FEATURES, 
                             out_channels=NUM_FEATURES,
                             kernel_size=3, stride=2, output_padding=1, 
                             padding=1),
          nn.BatchNorm2d(NUM_FEATURES),
          nn.LeakyReLU(),
          nn.ConvTranspose2d(in_channels=NUM_FEATURES, 
                             out_channels=NUM_FEATURES,
                             kernel_size=3, stride=2, output_padding=1, 
                             padding=1),
          nn.BatchNorm2d(NUM_FEATURES),
          nn.LeakyReLU(),
          nn.ConvTranspose2d(in_channels=NUM_FEATURES,
                             out_channels=3, 
                             kernel_size=3, stride=1, 
                             padding=1),
        )

    def forward(self, input_vec):
        output_vec = self.decoder(input_vec)
        return output_vec

class VariationalAutoencoder(nn.Module):
    def __init__(self, low_dim):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(low_dim)
        self.decoder = Decoder(low_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    
    
var_ae_model = VariationalAutoencoder(200).to(device)
print(var_ae_model)



# ------------------------------------------------------------
# --- Training or loading the model
# ------------------------------------------------------------

if TRAIN_MODEL:

    optimizer = torch.optim.Adam(var_ae_model.parameters())
    var_ae_model = var_ae_model.train()

    for epoch in range(N_EPOCHS):

        train_loss = 0

        for inputs, _ in celeb_dataloader:

            inputs = inputs.to(device)
            optimizer.zero_grad()
            estimated_outputs = var_ae_model(inputs)
            loss = ((estimated_outputs - inputs)**2).sum() + \
                var_ae_model.encoder.kl
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f"Epoch {epoch} : "
            f"train loss = {train_loss / len(celeb_dataloader.dataset)} ")

    # Save the model

    torch.save(var_ae_model.state_dict(), "models/var_ae_model_2.pth")
    
else:
    var_ae_model.load_state_dict(torch.load("models/var_ae_model_2.pth"))



# ------------------------------------------------------------
# --- See an image and its reconstruction
# ------------------------------------------------------------

var_ae_model = var_ae_model.eval()

plt.imshow(celeb_dataset[0][0].permute(1, 2, 0))
reconstr = var_ae_model(celeb_dataset[0][0].unsqueeze(0).to(device))
plt.imshow(reconstr.to('cpu').detach().squeeze().permute(1, 2, 0))



# ------------------------------------------------------------
# --- Interpolate between two images
# ------------------------------------------------------------

def interpolate(autoencoder, x_1, x_2, n=10):
    z_1 = autoencoder.encoder(x_1)
    z_2 = autoencoder.encoder(x_2)
    z = torch.concat([z_1 + (z_2 - z_1)*t for t in np.linspace(0, 1, n)])
    interpolate_list = autoencoder.decoder(z)
    interpolate_list = interpolate_list.to('cpu').detach().numpy()

    w = 32
    img = np.zeros((w, n*w, 3))
    for i, x_hat in enumerate(interpolate_list):
        img[:, i*w:(i+1)*w] = x_hat.squeeze().transpose(1, 2, 0)
    plt.figure(figsize = (40, 4))
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])

# Ids of the images to interpolate
id_1 = 1493
id_2 = 10344

plt.figure(figsize = (10, 2))
plt.imshow(celeb_dataset[id_1][0].permute(1, 2, 0))
plt.figure(figsize = (10, 2))
plt.imshow(celeb_dataset[id_2][0].permute(1, 2, 0))
interpolate(var_ae_model, celeb_dataset[id_1][0].unsqueeze(0).to(device), 
            celeb_dataset[id_2][0].unsqueeze(0).to(device))



# ------------------------------------------------------------
# --- Modify attributes to images
# ------------------------------------------------------------

# Load the attributes of the images
celeb_df = pd.read_csv("data/celeba-dataset/list_attr_celeba.csv")

# List of attributes
celeb_df.columns

# Define an attribute name
attribute = 'Male'

# A new class inheriting from ImageFolder giving the index of image
class ImageFolderWithIndex(ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithIndex, self).__getitem__(index)
        tuple_with_index = (index, original_tuple)
        return tuple_with_index
    
# Create the dataset with the index
celeb_idx_dataset = ImageFolderWithIndex(
    "data/celeba-dataset/img_align_celeba", 
    transform=transform_img)

# Unshuffled dataloader
unshuffled_dataloader = DataLoader(celeb_idx_dataset, batch_size=BATCH_SIZE, 
                                   shuffle=False)

# To store the sum of z values for positive and negative attributes
positive_sum_z = np.zeros(200)
negative_sum_z = np.zeros(200)
n_pos = 0
n_neg = 0

# Iterate over the dataset
for idx, (inputs, _) in unshuffled_dataloader:
    inputs = inputs.to(device)
    estimated_outputs = var_ae_model.encoder(inputs)
    is_pos = celeb_df[attribute].to_numpy()[idx]
    is_pos[is_pos == -1] = 0
    n_pos += is_pos.sum()
    n_neg += (1 - is_pos).sum()
    positive_sum_z += (estimated_outputs.to('cpu').detach().numpy() \
        * is_pos.reshape(-1, 1)).sum(0)
    negative_sum_z += (estimated_outputs.to('cpu').detach().numpy() \
        * (1 - is_pos).reshape(-1, 1)).sum(0)
    
# Compute mean of the z values for positive and negative attributes
positive_mean_z = positive_sum_z / n_pos
negative_mean_z = negative_sum_z / n_neg

# Get an image 
id_img = 102

# Display it
plt.figure(figsize = (10, 2))
plt.imshow(celeb_dataset[id_img][0].permute(1, 2, 0))

n_images = 9
max_span = 3

z_img = var_ae_model.encoder(celeb_dataset[id_img][0].unsqueeze(0).to(device))
p_z = torch.tensor(positive_mean_z).unsqueeze(0).to(device).to(torch.float32)
n_z = torch.tensor(negative_mean_z).unsqueeze(0).to(device).to(torch.float32)

z = torch.concat([z_img + (p_z - n_z)*t 
                  for t in np.linspace(-max_span, max_span, n_images)])
interpolate_list = var_ae_model.decoder(z)
interpolate_list = interpolate_list.to('cpu').detach().numpy()

w = 32
img = np.zeros((w, n_images*w, 3))
for i, x_hat in enumerate(interpolate_list):
    img[:, i*w:(i+1)*w] = x_hat.squeeze().transpose(1, 2, 0)
plt.figure(figsize = (40, 4))
plt.imshow(img)
plt.xticks([])
plt.yticks([])