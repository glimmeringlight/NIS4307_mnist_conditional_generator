import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import torchvision
from torchvision import transforms
from torch.utils import data
from torchvision.utils import save_image

# training params
batch_size = 64
g_optim_lr = 1e-4
d_optim_lr = 1e-5
epoch_time = 100
g_net_path = "../nets/Generator.pth"
d_net_path = "../nets/Discriminator.pth"
save_net = True


# 独热编码
# 输入x代表默认的torchvision返回的类比值，class_count类别值为10
def one_hot(x, class_count=10):
    return torch.eye(class_count)[x, :]  # 切片选取，第一维选取第x个，第二维全要


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
# why (0.5,0.5)? reference: https://blog.csdn.net/qq_42951560/article/details/114839052

dataset = torchvision.datasets.MNIST('../data',
                                     train=True,
                                     transform=transform,
                                     target_transform=one_hot,
                                     download=True)
dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
count = len(dataloader)


# 定义生成器，接受参数in_dim表示输入随机数的维度
class Generator(nn.Module):
    def __init__(self, in_dim):
        super(Generator, self).__init__()
        self.linear1 = nn.Linear(10, 128 * 7 * 7)
        self.bn1 = nn.BatchNorm1d(128 * 7 * 7)
        self.linear2 = nn.Linear(in_dim, 128 * 7 * 7)
        self.bn2 = nn.BatchNorm1d(128 * 7 * 7)
        self.deconv1 = nn.ConvTranspose2d(256, 128,
                                          kernel_size=(3, 3),
                                          padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128, 64,
                                          kernel_size=(4, 4),
                                          stride=2,
                                          padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv3 = nn.ConvTranspose2d(64, 1,
                                          kernel_size=(4, 4),
                                          stride=2,
                                          padding=1)

    def forward(self, x1, x2):
        # x1是label，x2是随机数
        x1 = F.relu(self.linear1(x1))
        x1 = self.bn1(x1)
        x1 = x1.view(-1, 128, 7, 7)
        x2 = F.relu(self.linear2(x2))
        x2 = self.bn2(x2)
        x2 = x2.view(-1, 128, 7, 7)
        x = torch.cat([x1, x2], axis=1)
        x = F.relu(self.deconv1(x))
        x = self.bn3(x)
        x = F.relu(self.deconv2(x))
        x = self.bn4(x)
        x = torch.tanh(self.deconv3(x))
        return x


# 定义判别器
# input:1，28，28的图片以及长度为10的condition
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.linear = nn.Linear(10, 1 * 28 * 28)
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.bn = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128 * 6 * 6, 1)  # 输出一个概率值

    def forward(self, x1, x2):
        # x1是条件张量，x2是输入图片
        x1 = F.leaky_relu(self.linear(x1))
        x1 = x1.view(-1, 1, 28, 28)
        x = torch.cat([x1, x2], axis=1)
        x = F.dropout2d(F.leaky_relu(self.conv1(x)))
        x = F.dropout2d(F.leaky_relu(self.conv2(x)))
        x = self.bn(x)
        x = x.view(-1, 128 * 6 * 6)
        x = torch.sigmoid(self.fc(x))
        return x


# 初始化模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
gen = Generator(100).to(device)
dis = Discriminator().to(device)
if os.path.exists(g_net_path):
    gen.load_state_dict(torch.load(g_net_path))
if os.path.exists(d_net_path):
    dis.load_state_dict(torch.load(d_net_path))

# 损失计算函数
loss_function = torch.nn.BCELoss()

# 定义优化器
d_optim = optim.Adam(dis.parameters(), lr=d_optim_lr)
g_optim = optim.Adam(gen.parameters(), lr=g_optim_lr)

# 产生随即噪音
noise_seed = torch.randn(batch_size, 100, device=device)
# 产生随机样本
label_seed = torch.randint(0, 10, size=(batch_size,))
print("This time label is {}".format(label_seed))
label_seed_onehot = one_hot(label_seed).to(device)

# 训练循环
for epoch in range(epoch_time):
    d_epoch_loss = 0
    g_epoch_loss = 0
    # 对全部的数据集做一次迭代
    for step, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        size = images.shape[0]
        random_noise = torch.randn(size, 100, device=device)

        # 训练判别器
        d_optim.zero_grad()
        real_output = dis(labels, images)
        d_real_loss = loss_function(real_output, torch.ones_like(real_output, device=device))
        d_real_loss.backward()

        gen_img = gen(labels, random_noise)
        fake_output = dis(labels, gen_img.detach())
        d_fake_loss = loss_function(fake_output, torch.zeros_like(fake_output, device=device))
        d_fake_loss.backward()

        d_loss = d_real_loss + d_fake_loss
        d_optim.step()

        # 训练生成器
        g_optim.zero_grad()
        fake_output = dis(labels, gen_img)
        g_loss = loss_function(fake_output, torch.ones_like(fake_output, device=device))
        g_loss.backward()
        g_optim.step()

        with torch.no_grad():
            d_epoch_loss += d_loss.item()
            g_epoch_loss += g_loss.item()
            if step % 100 == 0 or (step + 1) == count:
                print("Epoch: {}/{} | step: {}/{} | g_loss: {:4f} | d_loss: {:4f}".format(
                    epoch + 1, epoch_time, step, count, g_epoch_loss / (step + 1), d_epoch_loss / (step + 1)))

    # end for
    with torch.no_grad():
        # loss均值
        avg_d_loss = d_epoch_loss / count
        avg_g_loss = g_epoch_loss / count
        save_image(gen(label_seed_onehot, noise_seed), './results/{}_epoch.png'.format(epoch))
        print(
            "End epoch {}, d_loss: {:4f}, g_loss: {:4f}. Generated pics successfully!".format(epoch, avg_d_loss,
                                                                                              avg_g_loss))
if save_net:
    torch.save(obj=Generator.state_dict(gen), f=g_net_path)
    torch.save(obj=Discriminator.state_dict(dis), f=d_net_path)
    print("Training ended, net saved in {} and {}.".format(g_net_path, d_net_path))
else:
    print("Training ended, net not saved.")
