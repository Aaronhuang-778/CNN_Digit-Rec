# 代码创作者： Aaron（黄炜）
# 联系方式：aaronwei@buaa.edu.cn
# 开发时间： 2022/5/23 23:38
import torch
from torchvision import transforms  # pytorch 视觉库中提供了一些数据变换的接口
from torchvision import datasets  # pytorch 视觉库提供了加载数据集的接口
import torch.nn as nn


BATCH_SIZE = 64  # 由于使用批量训练的方法，需要定义每批的训练的样本数目
EPOCHS = 3  # 总共训练迭代的次数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate = 0.001  # 设定初始的学习率
drop = 0.6

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

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])),
    batch_size=BATCH_SIZE, shuffle=True)


net = Module()
net = torch.load('CNN0.6.model')
net.eval()
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        output = net(data.reshape(-1, 1, 28, 28))
        pred = output.data.max(1, keepdim=True)[1]  # 找到概率最大的下标，为输出值
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

print('\nAccuracy: {}/{} ({:.3f}%)\n'.format(correct, len(test_loader.dataset),
                                             100. * correct / len(test_loader.dataset)))