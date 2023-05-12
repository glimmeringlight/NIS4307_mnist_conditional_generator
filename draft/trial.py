import time

import torch
import torch.nn as nn
import torch.nn.functional as nn_fc
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image

# pre defined params
batch_size = 128
datasets = datasets.MNIST(root="../data/", transform=transforms.ToTensor(), download=False)
data_loader = DataLoader(dataset=datasets, batch_size=batch_size, shuffle=False)
loss_fc = nn.BCELoss()
lr = 0.002

# train params
training_epoch = 4


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # process image, in: [batch_size, 100, 1, 1]; out: [batch_size, 256, 4, 4]
        self.deConv_image = nn.Sequential(
            nn.ConvTranspose2d(100, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )

        # process classes, in: [batch_size, 10, 1, 1];out: [batch_size, 256, 4, 4]
        self.deConv_label = nn.Sequential(
            nn.ConvTranspose2d(10, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )

        # process both image and labels, in: [batch_size, 512, 4, 4]
        self.main_layer = nn.Sequential(
            # 512*4*4
            nn.ConvTranspose2d(512, 128, 3, 3, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 1, 3, 3, 1, bias=False),
            nn.Tanh()
            # 1*28*28
        )

    def forward(self, in_noise, in_label, batch_dim):
        # noise: [batch_size, 100]; label:[batch_size]
        # pre_process noise and label
        in_noise = torch.unsqueeze(in_noise, dim=2)
        in_noise = torch.unsqueeze(in_noise, dim=3)
        # now noise: [batch, 100, 1, 1]
        temp_label = torch.zeros(batch_dim, 10)
        for i in range(batch_dim):
            temp_label[i][in_label[i]] = 1
        in_label = temp_label
        # now label: [batch_size, 10]
        in_label = torch.unsqueeze(in_label, dim=2)
        in_label = torch.unsqueeze(in_label, dim=3)
        # now label: [batch_size, 10, 1, 1]
        in_noise = self.deConv_image(in_noise)
        in_label = self.deConv_label(in_label)
        noise_and_label = torch.cat((in_noise, in_label), dim=1)
        noise_and_label = self.main_layer(noise_and_label)
        return noise_and_label


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.process_image = nn.Sequential(
            # [batch, 1, 28, 28]
            # m=nn.Conv2d(1, 1, 1, padding=2)
            nn.Upsample(scale_factor=1.15),
            # [batch, 1, 32, 32]
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
            # [batch, 64, 16, 16]
        )
        self.process_label = nn.Sequential(
            # [batch, 10, 1, 1]
            nn.ConvTranspose2d(10, 64, 20, padding=2)
            # [batch, 64, 16, 16]
        )
        self.main_layer = nn.Sequential(
            # input: [batch, 128, 16, 16]
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. [batch, 256, 8, 8]
            # nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(512),
            # nn.LeakyReLU(0.2, inplace=True),
            # state size. [batch, 256, 4, 4]
            nn.Conv2d(256, 1, 6, 3, 0, bias=False),
            # state size. [batch, 1, 1, 1]
            nn.Sigmoid()
        )

    def forward(self, in_images, in_labels, batch_dim):
        # input: [batch_size, 1, 28, 28]; [batch_size]
        batch = in_images.shape[0]
        in_images = self.process_image(in_images)

        temp_labels = torch.zeros(batch_dim, 10)
        for i in range(batch_dim):
            temp_labels[i][in_labels[i]] = 1
        in_labels = temp_labels
        in_labels = torch.unsqueeze(in_labels, dim=2)
        in_labels = torch.unsqueeze(in_labels, dim=3)
        in_labels = self.process_label(in_labels)

        images_and_labels = torch.cat((in_images, in_labels), dim=1)
        prob = self.main_layer(images_and_labels)
        # labels: [batch, 1, 1, 1]
        prob = torch.reshape(prob, [batch])
        return prob


D = Discriminator()
G = Generator()
optim_D = optim.Adam(params=D.parameters(), lr=lr)
optim_G = optim.Adam(params=G.parameters(), lr=lr)
total_D_loss = 0
total_G_loss = 0

# @profile
def training():
    # 训练判别器
    real_prob_output = D(images, labels, batch_size)
    real_loss = loss_fc(real_prob_output, torch.ones_like(real_prob_output))

    fake_output = G(torch.randn(batch_size, 100), labels, batch_size)
    fake_prob_output = D(fake_output, labels, batch_size)
    fake_loss = loss_fc(fake_prob_output, torch.zeros_like(fake_prob_output))

    loss_d = real_loss + fake_loss
    # 记录损失
    remark_d_loss = loss_d

    optim_D.zero_grad()
    loss_d.backward()
    optim_D.step()

    # 训练生成器
    fake_output = G(torch.randn(batch_size, 100), labels, batch_size)
    fake_prob_output = D(fake_output, labels, batch_size)
    loss_g = loss_fc(fake_prob_output, torch.ones_like(fake_prob_output))

    optim_G.zero_grad()
    loss_g.backward()
    optim_G.step()

    return remark_d_loss, loss_g


for epoch in range(training_epoch):
    total_D_loss = 0
    total_D_loss = 0
    for index, (images, labels) in enumerate(data_loader):
        # print(index)
        loss_d_, loss_g_ = training()
        total_D_loss += loss_d_
        total_G_loss += loss_g_
        # 训练结束
        if (index + 1) % 10 == 0 or (index + 1) == len(data_loader):
            print('Epoch {:02d} | Step {:04d} / {} | Loss_D {:.4f} | Loss_G {:.4f}'.format(epoch, index + 1,
                                                                                           len(data_loader),
                                                                                           total_D_loss / (index + 1),
                                                                                           total_G_loss / (index + 1)))
        # exit()
        if index == 50:
            break

    # 该次迭代结束
    labels = torch.randint(0, 10, [batch_size])
    fake = G(torch.randn(batch_size, 100), labels, batch_size)
    save_image(fake, '../results/epoch_{}.png'.format(epoch))
