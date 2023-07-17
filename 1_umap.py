from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import umap
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset

mnist_dataset = MNIST("data/", train=True, download=True, transform=transforms.ToTensor())
mnist_dataloader = DataLoader(mnist_dataset, batch_size=64, shuffle=True)

# UMAP plot 

reducer = umap.UMAP()
embedding = reducer.fit_transform(mnist_dataset.data.flatten(1))

sns.set(context="paper", style="white")
fig, ax = plt.subplots(figsize=(12, 10))
color = mnist_dataset.targets
plt.scatter(embedding[:, 0], embedding[:, 1], c=color, cmap="Spectral", s=0.1)
plt.setp(ax, xticks=[], yticks=[])
plt.title("MNIST data embedded into two dimensions by UMAP", fontsize=18)
plt.show()