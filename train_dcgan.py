# https://debuggercafe.com/implementing-deep-convolutional-gan-with-pytorch/
import torch, torchvision
import torch.nn as nn 
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from utils import save_generator_image, weight_init
from utils import label_fake, label_real, create_noise
from model import Generator, Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

matplotlib.style.use("ggplot")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train():
    image_size=64
    batch_size=128
    nz=100
    beta1=0.5 
    lr = 2e-4
    sample_size=64
    epochs=25

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_data = torchvision.datasets.CIFAR10("../data", train=True, download=False, transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    gen = Generator(nz).to(device)
    dis = Discriminator().to(device)

    gen.apply(weight_init)
    dis.apply(weight_init)

    print('##### GENERATOR #####')
    print(gen)
    print('######################')
    print('\n##### DISCRIMINATOR #####')
    print(dis)
    print('######################')

    optim_g = optim.Adam(gen.parameters(), lr=lr, betas=(beta1, 0.999))
    optim_d = optim.Adam(dis.parameters(), lr=lr, betas=(beta1, 0.999))

    criterion = nn.BCELoss()

    losses_g = []
    losses_d = []

    def train_dis(optim, data_real, data_fake):
        b_size = data_real.size(0)
        real_label = label_real(b_size)
        fake_label = label_fake(b_size)

        optim.zero_grad()

        out_real = dis(data_real).view(-1)
        loss_real = criterion(out_real, real_label)
        out_fake = dis(data_fake)
        loss_fake = criterion(out_fake, fake_label)

        loss_real.backward()
        loss_fake.backward()
        optim.step()

        return loss_real + loss_fake

    def train_gen(optim, data_fake):
        b_size = data_fake.size(0)
        real_label = label_real(b_size)
        
        optim.zero_grad()

        out = dis(data_fake)
        loss = criterion(out, real_label)

        loss.backward()
        optim.step()

        return loss

    noise = create_noise(sample_size, nz)

    gen.train()
    dis.train()

    writer = SummaryWriter()

    for epoch in range(epochs):
        loss_g = 0.0
        loss_d = 0.0
        for bi, data in tqdm(enumerate(train_loader), total=int(len(train_data) / train_loader.batch_size)):
            image, _ = data
            image = image.to(device)
            b_size = len(image)
            data_fake = gen(create_noise(b_size, nz)).detach()
            data_real = image
            loss_d += train_dis(optim_d, data_real, data_fake)
            data_fake = gen(create_noise(b_size, nz))
            loss_g += train_gen(optim_g, data_fake)
        generated_image = gen(noise).cpu().detach()
        save_generator_image(generated_image, f"./output/gen_img{epoch}.png")
        epoch_loss_g = loss_g / bi
        epoch_loss_d = loss_d / bi
        losses_g.append(epoch_loss_g)
        losses_d.append(epoch_loss_d)
        writer.add_scalar("Loss/Gen", loss_g, epoch)

        print(f"Epoch {epoch+1} of epochs")
        print(f"Generator loss: {epoch_loss_g:.8f}, Discriminator Loss: {epoch_loss_d:.8f}")

    print("Done training")
    torch.save(gen.state_dict(), './model/gen.pth')
    plt.figure()
    plt.plot(loss_g, label="Generator loss")
    plt.plot(loss_d, label="Discriminator loss")
    plt.legend()
    plt.savefig('../output/loss.png')


if __name__ == '__main__':
    train()
