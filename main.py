import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torchvision.utils import save_image
from torchvision import datasets
from PIL import UnidentifiedImageError
from PIL import Image
import os
import os.path
import torch.utils.data as data
import torchvision.transforms as transforms
import open_clip
import numpy as np


# Load the CLIP model and tokenizer
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# Set the device to use
device = "cuda" if torch.cuda.is_available() else "cpu"

# Move the model to the selected device and set it to evaluation mode
clip_model = model.to(device)
clip_model.eval()

from torch.utils.data import Dataset


def default_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class ImageFolderWithPaths(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        super(ImageFolderWithPaths, self).__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.imgs = self._load_imgs()

    def _load_imgs(self):
        imgs = []
        for root, _, files in os.walk(self.root):
            for f in files:
                path = os.path.join(root, f)
                item = (path, os.path.basename(os.path.dirname(path)))
                imgs.append(item)
        return imgs

    def __getitem__(self, index):
        path, label = self.imgs[index]
        try:
            img = Image.open(path).convert('RGB')
        except (FileNotFoundError, OSError, TypeError, ValueError, IndexError, UnidentifiedImageError):
            print(f"Skipping {path} because img is invalid.")
            return None, None, None

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label, path
    def __len__(self):
        return len(self.imgs)
        
class Generator(nn.Module):
    def __init__(self, z_dim, img_channels, features_gen):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            self._block(z_dim, features_gen * 8, 4, 1, 0),
            self._block(features_gen * 8, features_gen * 4, 4, 2, 1),
            self._block(features_gen * 4, features_gen * 2, 4, 2, 1),
            self._block(features_gen * 2, features_gen, 4, 2, 1),
            nn.ConvTranspose2d(features_gen, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.gen(x)

class Discriminator(nn.Module):
    def __init__(self, img_channels, features_disc):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            self._block(img_channels, features_disc, 4, 2, 1),
            self._block(features_disc, features_disc * 2, 4, 2, 1),
            self._block(features_disc * 2, features_disc * 4, 4, 2, 1),
            self._block(features_disc * 4, features_disc * 8, 4, 2, 1),
            nn.Conv2d(features_disc * 8, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.disc(x)
    
# Hyperparameters
img_size = 64
img_channels = 3
z_dim = 100 # Dimension of noise to Generator
features_gen = 64   # Number of features Generator outputs
features_disc = 64 # Number of features Discriminators inputs
batch_size = 128 # How many images per batch
lr = 0.0002 # Learning rate
beta1 = 0.5 # Coefficients used for computing running averages of gradient and its square
beta2 = 0.999 # Constant 2 for optimizer
num_epochs = 100 # Number of generations the model will go through
clip_weight = 1.0  # Weight for the CLIP-based loss

def clip_loss(images, prompt_text):
    # Normalize the images to the range [-1, 1] and preprocess them
    images = (images + 1) / 2 
    images_np = (images.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype(np.uint8)
    images_pil = Image.fromarray(np.concatenate(images_np, axis=0))
    images_clip = preprocess(images_pil)

    # Move the images to the device
    images_clip = images_clip.to(device)

    # Encode the images and text using the CLIP model
    with torch.no_grad():
        image_features = clip_model.encode_image(images_clip)
        text_tokens = tokenizer([prompt_text] * images.size(0)).to(device)
        text_features = clip_model.encode_text(text_tokens)

    # Calculate the similarity between image and text features
    similarity = torch.matmul(image_features, text_features.t()).diagonal()

    # Return the negative similarity as the loss
    return -similarity.mean()




# Set the loss function for the discriminator
criterion = nn.BCELoss()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the generator and discriminator
gen = Generator(z_dim, img_channels, features_gen).to(device)
disc = Discriminator(img_channels, features_disc).to(device)

# Initialize the weights for the generator and discriminator
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02) 
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

gen.apply(weights_init)
disc.apply(weights_init)

# Set the optimizers for the generator and discriminator
optimizer_gen = optim.Adam(gen.parameters(), lr=lr, betas=(beta1, beta2))
optimizer_disc = optim.Adam(disc.parameters(), lr=lr, betas=(beta1, beta2))

preprocess = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = ImageFolderWithPaths("wikiart", transform=preprocess)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)


# Training loop
for epoch in range(num_epochs):
    for i, (real_imgs, _, paths) in enumerate(dataloader):
        # if real_imgs is None or paths is None or None in real_imgs or None in paths:
        #     continue
        if any(x is None for x in real_imgs) or any(not isinstance(x, str) for x in paths) or not all(paths):
            continue
        real_imgs = real_imgs.to(device)
        batch_size = real_imgs.size(0)
        real_labels = torch.ones(batch_size, 1, 1, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1, 1, 1).to(device)
        # Train the discriminator
        optimizer_disc.zero_grad()

        # Train on real images
        real_outputs = disc(real_imgs)
        real_loss = criterion(real_outputs, real_labels)

        # Train on fake images
        z = torch.randn(batch_size, z_dim, 1, 1).to(device)
        fake_imgs = gen(z)
        fake_outputs = disc(fake_imgs.detach())
        fake_loss = criterion(fake_outputs, fake_labels)

        # Combine real and fake losses and update the discriminator
        disc_loss = real_loss + fake_loss
        disc_loss.backward()
        optimizer_disc.step()

        # Train the generator
        optimizer_gen.zero_grad()

        # Generate images and calculate the adversarial loss
        fake_outputs = disc(fake_imgs)
        gen_loss = criterion(fake_outputs, real_labels)

        # Calculate the CLIP-based loss
        prompt_text = os.path.splitext(os.path.basename(list(paths)[0]))[0]

        #clip_loss_value = clip_loss(fake_imgs, prompt_text) * clip_weight

        # Combine adversarial and CLIP losses and update the generator
        #total_gen_loss = gen_loss + clip_loss_value
        total_gen_loss = gen_loss
        total_gen_loss.backward()
        optimizer_gen.step()

        # Print the losses
        if i % 50 == 0:
            # print(f"Epoch [{epoch}/{num_epochs}] Batch {i}/{len(dataloader)} \
            #         Loss D: {disc_loss:.4f}, Loss G: {gen_loss:.4f}, Loss G (CLIP): {clip_loss_value:.4f}")
            print(f"Epoch [{epoch}/{num_epochs}] Batch {i}/{len(dataloader)} \
                    Loss D: {disc_loss:.4f}, Loss G: {gen_loss:.4f}")

    # Save the generated images and the generator model periodically
    if epoch % 10 == 0:
        save_image(fake_imgs, f"images/fake_images_{epoch}.png", normalize=True)
        torch.save(gen.state_dict(), f"models/generator_{epoch}.pth")