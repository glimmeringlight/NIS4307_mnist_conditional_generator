import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils import data
from torchvision.utils import save_image


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

    # forward method
    def forward(self, input, label):
        x = F.relu(self.deconv1_1_bn(self.deconv1_1(input)))
        y = F.relu(self.deconv1_2_bn(self.deconv1_2(label)))
        x = torch.cat([x, y], 1)
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = torch.tanh(self.deconv4(x))

        return x


class AiGcMn:
    def __init__(self):
        # 初始化生成器
        self.gen = Generator()
        self.gen_path = './nets/generator.pth'
        self.gen.load_state_dict(torch.load(self.gen_path))
        self.resize = transforms.Resize(28)

        # 初始化独热编码表
        onehot = torch.zeros(10, 10)
        self.onehot = onehot.scatter_(1, torch.LongTensor(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10, 1), 1).view(10, 10, 1, 1)

    def encode_1hot(self, labels):
        # 1hot for labels
        labels_1hot = self.onehot[labels.type(torch.LongTensor)]
        return labels_1hot

    def generate(self, labels):
        batch_size = labels.shape[0]
        # 独热编码
        labels_1hot = self.encode_1hot(labels)
        # random noise
        batch_size = labels.shape[0]
        rd_noise = torch.randn((batch_size, 100)).view(-1, 100, 1, 1)
        # 生成图片
        gen_output = self.gen(rd_noise, labels_1hot)
        # 处理为28*28
        processed_output = torch.zeros(batch_size, 1, 28, 28)
        for i in range(batch_size):
            processed_output[i] = self.resize(gen_output[i])
        return processed_output


if __name__ == '__main__':
    aigcmn = AiGcMn()
    labels = [1, 1, 4, 5, 1, 4]
    labels = torch.Tensor(labels)
    gen_output = aigcmn.generate(labels)

    # 保存tensor
    gen_output_numpy = gen_output.detach().numpy()
    gen_output_numpy.tofile('./output/tensor.csv', sep=',')
    print("Saved csv file!")

    # 保存图片
    save_image(gen_output, 'output/output.png')
    print("Successfully saved output.")
