# 代码创作者： Aaron（黄炜）
# 联系方式：aaronwei@buaa.edu.cn
# 开发时间： 2022/5/23 21:05
import time

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler  # 学习率调整器，在训练过程中合理变动学习率
from torchvision import transforms  # pytorch 视觉库中提供了一些数据变换的接口
from torchvision import datasets  # pytorch 视觉库提供了加载数据集的接口

# 预设网络超参数
BATCH_SIZE = 64  # 由于使用批量训练的方法，需要定义每批的训练的样本数目
EPOCHS = 3  # 总共训练迭代的次数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate = 0.001  # 设定初始的学习率
drop = 0.6
# 加载训练集
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize(mean=(0.5,), std=(0.5,))  # 数据规范化到正态分布
                   ])),
    batch_size=BATCH_SIZE, shuffle=True)  # 指明批量大小，打乱，这是处于后续训练的需要。

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])),
    batch_size=BATCH_SIZE, shuffle=True)

class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        # 学习层
        # 共有四层
        self.first = nn.Sequential(
            # 前两层特征层
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            # 池化卷积层
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 分类层
        self.classifier = nn.Sequential(
            nn.Dropout(p=drop),
            # 这里是通道数64 * 图像大小7 * 7，然后输入到512个神经元中
            nn.Linear(64 * 7 * 7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop),
            nn.Linear(512, 10),
        )
    def forward(self, x):
        x = self.first(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def train(_epoch, model, device, train, optimizer, lr_sch):
    model.train()
    lr_sch.step()
    for epoch in range(_epoch):
        for i, (images, labels) in enumerate(train):
            samples = images.to(device)
            labels = labels.to(device)
            input = model(samples.reshape(0-1, 1, 28, 28))
            loss1 = loss(input, labels)
            # 优化器内部的参数梯度调整为0
            optimizer.zero_grad()
            loss1.backward()
            optimizer.step()
            if (i+1) % 100 == 0:
                print("Epoch:{}/{}, step:{}, loss:{:.4f}".format(epoch + 1, _epoch, i + 1, loss1.item()))


def test(_test_loader, _model, _device):
    _model.eval()  # 设置模型进入预测模式 evaluation
    loss1 = 0
    correct = 0

    with torch.no_grad():  # 如果不需要 backward更新梯度，那么就要禁用梯度计算，减少内存和计算资源浪费。
        for data, target in _test_loader:
            data, target = data.to(_device), target.to(_device)
            output = module(data.reshape(-1, 1, 28, 28))
            loss1 += loss(output, target).item()  # 添加损失值
            pred = output.data.max(1, keepdim=True)[1]  # 找到概率最大的下标，为输出值
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()  # .cpu()是将参数迁移到cpu上来。

    loss1 /= len(_test_loader.dataset)

    print('\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        loss1, correct, len(_test_loader.dataset),
        100. * correct / len(_test_loader.dataset)))

if __name__ == '__main__':
    start = time.time()
    model_path = "svm_v4.model"
    test_ubyte_path = "dataset/t10k-images.idx3-ubyte"
    label_path = "dataset/t10k-labels.idx1-ubyte"
    test_img_path = "test_image/"
    module = Module()
    module = module.to(DEVICE)
    loss = nn.CrossEntropyLoss()
    loss = loss.to(DEVICE)
    optimizer = torch.optim.Adam(module.parameters(), lr=learning_rate)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)

    for epoch in range(1, EPOCHS + 1):
        train(epoch, module, DEVICE, train_loader, optimizer, exp_lr_scheduler)
        test(test_loader, module, DEVICE)
        test(train_loader, module, DEVICE)

    torch.save(module, 'CNN0.6.model')
    print("Time: " + str(time.time() - start))