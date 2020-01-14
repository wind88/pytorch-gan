# encoding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 392),
            nn.ReLU(),
            nn.Linear(392, 784),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 392),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(392, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(x.shape[0], 784)
        return self.net(x)


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = datasets.MNIST('./data/', transform=transform, download=True)
    dataload = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)

    # 生成器，判断器
    gen_net = Generator()
    dis_net = Discriminator()

    # 定义损失及优化器
    criterion = nn.BCELoss()
    optim_g = torch.optim.Adam(gen_net.parameters(), lr=0.0001)
    optim_d = torch.optim.Adam(dis_net.parameters(), lr=0.0001)

    for epoch in range(300):
        for i, (inputs, labels) in enumerate(dataload):
            batch_size = inputs.shape[0]
            inputs = inputs.to(device)

            # 判别器判别真实用例
            dis_net.zero_grad()
            real_label = torch.ones(batch_size).to(device)
            real_out = dis_net(inputs.reshape(batch_size, -1)).view(-1)
            real_loss = criterion(real_out, real_label)
            real_loss.backward()  # 计算真样本梯度

            # 生成器生成图片
            noise = torch.autograd.Variable(torch.FloatTensor(torch.randn(batch_size, 128))).to(device)
            g_noise = gen_net(noise)
            noise_label = torch.zeros(batch_size).to(device)

            # 使用判别器判别假图片
            noise_out = dis_net(g_noise).view(-1)
            noise_loss = criterion(noise_out, noise_label)
            noise_loss.backward(retain_graph=True)  # 计算假样本被判断为假样本梯度
            dis_loss = real_loss + noise_loss
            optim_d.step()

            # 生成器梯度更新
            gen_net.zero_grad()
            noise_out = dis_net(g_noise).view(-1)
            gen_loss = criterion(noise_out, real_label)
            gen_loss.backward(retain_graph=True)  # 计算假样本被判断为真样本梯度
            optim_g.step()

            print('Epoch: {}, batch: {}, d_loss: {}, g_loss: {}'.format(epoch, i, dis_loss, gen_loss))

        # 生成图片
        if (epoch+1) % 50 == 0:
            img_data = np.zeros([28 * 8, 28 * 8], dtype=np.float)
            # print(g_noise.shape)
            g_noise = torch.autograd.Variable(g_noise)
            for j in range(64):
                img = g_noise[j].reshape(28, 28)
                if j % 8 == 0:
                    i_start = (j // 8) * 28
                j_start = (j % 8) * 28
                img_data[i_start:i_start+28:, j_start:j_start+28] = img
            plt.switch_backend('agg')
            plt.gray()
            plt.imsave('./data/'+str(epoch+1)+'.png', img_data)
