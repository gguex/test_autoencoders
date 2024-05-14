import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# ------------------------------------------------------------
# --- Constants
# ------------------------------------------------------------

IMAGE_SIZE = 64
CHANNELS = 3
BATCH_SIZE = 32
LATENT_DIM = 32
CRITIC_STEPS = 3
LEARNING_RATE = 0.00005
ADAM_BETA_1 = 0.5
ADAM_BETA_2 = 0.9
GP_WEIGHT = 10.0
EPOCHS = 5

TRAIN_MODEL = True

# ------------------------------------------------------------
# --- Dataset
# ------------------------------------------------------------

# A new class inheriting from ImageFolder that returns 
# the image one hot vectors representing the features
class ImageFolderWithAttributes(ImageFolder):
    
    def __init__(self, path_to_attributes, **kwargs):
        super().__init__(**kwargs)
        attributes_df = pd.read_csv("data/celeba-dataset/list_attr_celeba.csv")
        attributes_tensor = torch.tensor(attributes_df.iloc[:, 1:].to_numpy()
                                         .astype('float32'))
        attributes_tensor[attributes_tensor == -1] = 0
        self.attributes = attributes_tensor
        self.n_attributes = attributes_tensor.size(1)
    
    def __getitem__(self, index):
        img = super().__getitem__(index)[0]
        feature = self.attributes[index, :]
        return img, feature
    
# The transformation function for the images
def transform_img(image):
    return transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                               transforms.ToTensor()])(image)

# The dataset
dataset = ImageFolderWithAttributes(
    root='data/celeba-dataset/img_align_celeba',
    path_to_attributes="data/celeba-dataset/list_attr_celeba.csv",
    transform=transform_img)

# The data loader
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ------------------------------------------------------------
# --- Model definition
# ------------------------------------------------------------

# Define the Generator network
class Generator(nn.Module):
    def __init__(self, latent_dim, n_attributes):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Unflatten(1, (latent_dim + n_attributes, 1, 1)),
            nn.ConvTranspose2d(in_channels=latent_dim + n_attributes, 
                               out_channels=128,
                               kernel_size=4, stride=1, bias=False),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(in_channels=128, 
                               out_channels=128,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(in_channels=128, 
                               out_channels=128,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(in_channels=128, 
                               out_channels=128,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(in_channels=128, 
                               out_channels=CHANNELS,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z, attributes):
        concat_vec = torch.concat([z, attributes], dim=1)
        img = self.model(concat_vec)
        return img

# Testing the Generator
my_gen = Generator(LATENT_DIM, dataset.n_attributes)
vector_input = torch.tensor(np.random.randn(1, LATENT_DIM)).to(torch.float32)
attribute_input = dataset[0][1].unsqueeze(0)
my_gen(vector_input, attribute_input).shape

# Define the Critic network
class Critic(nn.Module):
    def __init__(self, n_attributes):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=CHANNELS + n_attributes,
                      out_channels=64, 
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=64, out_channels=128, 
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Conv2d(in_channels=128, out_channels=128, 
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Conv2d(in_channels=128, out_channels=128, 
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Conv2d(in_channels=128, out_channels=1, 
                      kernel_size=4, stride=1, padding=0),
            nn.Flatten(1, -1)
        )
        
    def forward(self, img, attributes):
        attributes_reshape = \
            attributes.unsqueeze(1).reshape(img.shape[0], 1, 1, -1) \
                .repeat((1, IMAGE_SIZE, IMAGE_SIZE, 1)) \
                    .permute(0, 3, 1, 2)
        concat_input = torch.concat([img, attributes_reshape], dim=1)
        validity = self.model(concat_input)
        return validity

# Testing the Critic
my_crit = Critic(dataset.n_attributes)
img_input = dataset[0][0].unsqueeze(0)
attribute_input = dataset[0][1].unsqueeze(0)
my_crit(img_input, attribute_input).shape



# ------------------------------------------------------------
# --- Model training
# ------------------------------------------------------------

# Create optimizers
optim_gen = optim.Adam(my_gen.parameters(), lr=LEARNING_RATE, 
                       betas=(ADAM_BETA_1, ADAM_BETA_2))
optim_crit = optim.Adam(my_crit.parameters(), lr=LEARNING_RATE, 
                        betas=(ADAM_BETA_1, ADAM_BETA_2))

my_crit = my_crit.to(device)
my_gen = my_gen.to(device)

if TRAIN_MODEL:
    
    my_crit.train()
    my_gen.eval()
    
    # Epochs loop
    for epoch in range(EPOCHS):
        
        # Init losses
        c_losses = []
        g_losses = []
        
        # Loop on dataset
        for img, attributes in dataloader:
            
            # Set attributes and images to device
            img = img.to(device)
            attributes = attributes.to(device)
    
            # Train the critic
            for i in range(CRITIC_STEPS):
                
                # Generate fake images
                z = torch.randn(attributes.size(0),
                                LATENT_DIM).to(device).to(torch.float32)
                gen_imgs = my_gen(z, attributes)
                
                # Compute answers
                critic_real = my_crit(img, attributes)
                critic_fake = my_crit(gen_imgs, attributes)
                
                # Compute the wassterstein loss
                c_wass_loss = torch.mean(critic_fake) - torch.mean(critic_real)
                
                # Compute gradient penalty
                e = torch.rand(BATCH_SIZE, 1, 1, 1).to(device)
                interpolates = (e * img.data + \
                                (1 - e) * gen_imgs.data).requires_grad_(True)
                d_interpolates = my_crit(interpolates, attributes)
                grad_outputs = torch.ones(d_interpolates.size()).to(device)
                grad = torch.autograd.grad(outputs=d_interpolates,
                                           inputs=interpolates,
                                           grad_outputs=grad_outputs,
                                           create_graph=True, retain_graph=True,
                                           only_inputs=True)[0]
                grad_norm = grad.view(grad.size(0), -1).norm(p=2, dim=1)
                c_gp = ((grad_norm - 1)**2).mean()
                
                # Compute total loss and update weights
                c_loss = c_wass_loss + GP_WEIGHT * c_gp
                my_crit.zero_grad()
                c_loss.backward()
                optim_crit.step()
                c_losses.append(c_loss.item())
            
            # --- Train the generator
            
            # Generate fake images
            z = torch.randn(attributes.size(0), 
                            LATENT_DIM).to(device).to(torch.float32)
            gen_imgs = my_gen(z, attributes)
            critic_fake = my_crit(gen_imgs, attributes)
            
            # Compute the generator loss and update weights
            g_loss = -torch.mean(critic_fake)
            my_gen.zero_grad()
            g_loss.backward()
            optim_gen.step()
            g_losses.append(g_loss.item())
            
        # Print losses
        print(f"Epoch {epoch+1}/{EPOCHS}, "
              f"Critic loss: {np.mean(c_losses)}, "
              f"Generator loss: {np.mean(g_losses)}")
        
    # Save the models
    torch.save(my_gen.state_dict(), "models/cgan_gen.pth")
    torch.save(my_crit.state_dict(), "models/cgan_crit.pth")