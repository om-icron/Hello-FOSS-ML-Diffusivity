import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.utils as vutils

class DCGenerator(nn.Module):
    def __init__(self, latent_dim, img_channels, feature_g):
        super(DCGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, feature_g * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_g * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_g * 8, feature_g * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_g * 4, feature_g * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_g * 2, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

class DCDiscriminator(nn.Module):
    def __init__(self, img_channels, feature_d):
        super(DCDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(img_channels, feature_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_d, feature_d * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_d * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_d * 2, feature_d * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_d * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_d * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img).view(-1, 1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 100
img_channels = 1
feature_g = 64
feature_d = 64
batch_size = 128
epochs = 50
lr = 0.0002
beta1 = 0.5

generator = DCGenerator(latent_dim, img_channels, feature_g).to(device)
discriminator = DCDiscriminator(img_channels, feature_d).to(device)

criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

images_path = 'Path to Dataset'
labels_path = 'Path to Dataset'

dataset = MNISTUbyteDataset(images_path, labels_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    for i, (imgs, _) in enumerate(dataloader):
        imgs = imgs.to(device)
        batch_size = imgs.size(0)
        real = torch.ones(batch_size, 1, device=device)
        fake = torch.zeros(batch_size, 1, device=device)

        optimizer_d.zero_grad()
        real_loss = criterion(discriminator(imgs), real)
        z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        fake_loss = criterion(discriminator(generator(z).detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_d.step()

        optimizer_g.zero_grad()
        g_loss = criterion(discriminator(generator(z)), real)
        g_loss.backward()
        optimizer_g.step()

        if i % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{i}/{len(dataloader)}], "
                  f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

    vutils.save_image(generator(z), f"generated_epoch_{epoch+1}.png", normalize=True)