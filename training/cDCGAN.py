import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils import data
from torchvision.utils import save_image


# G(z)
class Generator(nn.Module):
    def __init__(self, d=128):
        super(Generator, self).__init__()
        self.deconv1_1 = nn.ConvTranspose2d(100, d * 2, 4, 1, 0)
        self.deconv1_1_bn = nn.BatchNorm2d(d * 2)
        self.deconv1_2 = nn.ConvTranspose2d(10, d * 2, 4, 1, 0)
        self.deconv1_2_bn = nn.BatchNorm2d(d * 2)
        self.deconv2 = nn.ConvTranspose2d(d * 4, d * 2, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d * 2)
        self.deconv3 = nn.ConvTranspose2d(d * 2, d, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d)
        self.deconv4 = nn.ConvTranspose2d(d, 1, 4, 2, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        x = F.relu(self.deconv1_1_bn(self.deconv1_1(input)))
        y = F.relu(self.deconv1_2_bn(self.deconv1_2(label)))
        x = torch.cat([x, y], 1)
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = torch.tanh(self.deconv4(x))

        return x


# D(G(z))
class Discriminator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(Discriminator, self).__init__()
        self.conv1_1 = nn.Conv2d(1, int(d / 2), 4, 2, 1)
        self.conv1_2 = nn.Conv2d(10, int(d / 2), 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, 1, 4, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        x = F.leaky_relu(self.conv1_1(input), 0.2)
        y = F.leaky_relu(self.conv1_2(label), 0.2)
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = torch.sigmoid(self.conv4(x))

        return x


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


# global settings
# training device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# training parameters
batch_size = 128
lr = 0.0002
train_epoch = 20
save_net = True
g_net_path = './nets/generator.pth'
d_net_path = './nets/discriminator.pth'
# data_loader
img_size = 32
resize = transforms.Resize(img_size)
transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
dataloader = data.DataLoader(datasets.MNIST('../data',
                                            train=True,
                                            transform=transform,
                                            download=True), batch_size=batch_size, shuffle=True)
dataloader_len = len(dataloader)

# loss function
# Binary Cross Entropy loss
BCE_loss = nn.BCELoss()

# 1hot
onehot = torch.zeros(10, 10)
onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10, 1), 1).view(10, 10, 1, 1).to(
    device)

fill = torch.zeros([10, 10, img_size, img_size])
for i in range(10):
    fill[i, i, :, :] = 1
fill = fill.to(device)

# fixed noise & label
# prepare z
temp_z_ = torch.randn(10, 100)
fixed_z_ = temp_z_
for i in range(9):
    fixed_z_ = torch.cat([fixed_z_, temp_z_], 0)
fixed_z_ = fixed_z_.view(-1, 100, 1, 1)
fixed_z_ = fixed_z_.to(device)

# prepare y
fixed_y_ = torch.zeros(10, 1)
for i in range(9):
    temp = torch.ones(10, 1) + i
    fixed_y_ = torch.cat([fixed_y_, temp], 0)
fixed_y_ = onehot[fixed_y_.type(torch.LongTensor)].squeeze()
fixed_y_ = fixed_y_.view(-1, 10, 1, 1)
fixed_y_ = fixed_y_.to(device)

# ready for training
# network
G = Generator(128)
D = Discriminator(128)
G = G.to(device)
D = D.to(device)

if os.path.exists(g_net_path):
    G.load_state_dict(torch.load(g_net_path))
else:
    G.weight_init(mean=0.0, std=0.02)
if os.path.exists(d_net_path):
    D.load_state_dict(torch.load(d_net_path))
else:
    D.weight_init(mean=0.0, std=0.02)

# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

print('training start!')
start_time = time.time()
for epoch in range(train_epoch):
    # learning rate decay
    if (epoch + 1) == 11:
        G_optimizer.param_groups[0]['lr'] /= 10
        D_optimizer.param_groups[0]['lr'] /= 10
        print("learning rate change!")

    if (epoch + 1) == 16:
        G_optimizer.param_groups[0]['lr'] /= 10
        D_optimizer.param_groups[0]['lr'] /= 10
        print("learning rate change!")

    epoch_start_time = time.time()
    y_real_ = torch.ones(batch_size).to(device)
    y_fake_ = torch.zeros(batch_size).to(device)

    # static info
    g_loss_total = 0
    d_loss_total = 0

    for step, (x_, y_) in enumerate(dataloader):
        # train discriminator D
        D.zero_grad()

        mini_batch = x_.size()[0]

        # 避免一组训练最后一次的batch_size不足预先设定的batch_size
        if mini_batch != batch_size:
            y_real_ = torch.ones(mini_batch)
            y_fake_ = torch.zeros(mini_batch)
            y_real_, y_fake_ = y_real_.to(device), y_fake_.to(device)

        y_fill_ = fill[y_]
        x_, y_fill_ = x_.to(device), y_fill_.to(device)

        D_result = D(x_, y_fill_).squeeze()
        D_real_loss = BCE_loss(D_result, y_real_)

        z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
        y_ = (torch.rand(mini_batch, 1) * 10).type(torch.LongTensor).squeeze()
        y_label_ = onehot[y_]
        y_fill_ = fill[y_]
        z_, y_label_, y_fill_ = z_.to(device), y_label_.to(device), y_fill_.to(device)

        G_result = G(z_, y_label_)
        D_result = D(G_result, y_fill_).squeeze()
        D_fake_loss = BCE_loss(D_result, y_fake_)

        D_train_loss = D_real_loss + D_fake_loss

        D_train_loss.backward()
        D_optimizer.step()

        # update static info
        with torch.no_grad():
            d_loss_total += D_train_loss.item()

        # train generator G
        G.zero_grad()

        z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
        y_ = (torch.rand(mini_batch, 1) * 10).type(torch.LongTensor).squeeze()
        y_label_ = onehot[y_]
        y_fill_ = fill[y_]
        z_, y_label_, y_fill_ = z_.to(device), y_label_.to(device), y_fill_.to(device)

        G_result = G(z_, y_label_)
        D_result = D(G_result, y_fill_).squeeze()

        G_train_loss = BCE_loss(D_result, y_real_)

        G_train_loss.backward()
        G_optimizer.step()

        # update static info
        with torch.no_grad():
            g_loss_total += G_train_loss.item()

            if (step + 1) % 100 == 0 or (step + 1) == dataloader_len:
                print("Epoch: {}/{} | step: {}/{} | avg_g_loss: {} | avg_d_loss: {}".format(
                    epoch + 1, train_epoch, step + 1, dataloader_len, g_loss_total / (step + 1),
                    d_loss_total / (step + 1)
                ))

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time

    print("Ended epoch {} | using time: {} | avg_g_loss: {} | avg_d_loss: {}".format(
        epoch + 1, per_epoch_ptime, g_loss_total / dataloader_len, d_loss_total / dataloader_len
    ))
    # generate output
    with torch.no_grad():
        generator_output = G(fixed_z_, fixed_y_)
        gen_resize = transforms.Resize(28)
        processed_output = torch.zeros(fixed_y_.shape[0], 1, 28, 28)
        for i in range(fixed_y_.shape[0]):
            processed_output[i] = gen_resize(generator_output[i])
        save_image(processed_output, './output/{}_epoch.png'.format(epoch + 1))

if save_net:
    torch.save(obj=Generator.state_dict(G), f=g_net_path)
    torch.save(obj=Discriminator.state_dict(D), f=d_net_path)
    print("Training ended, net saved in {} and {}.".format(g_net_path, d_net_path))
else:
    print("Training ended, net not saved.")
