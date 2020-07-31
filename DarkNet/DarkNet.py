import torch
import torch.nn as nn
import math
from collections import OrderedDict


class ResiduleBlock(nn.Module):
    def __init__(self,in_channle,middle_channle):
        super(ResiduleBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_channle,middle_channle,kernel_size=1,stride=1,padding=0,bias=False)  #不能除以，必须是整数吧
        self.bn1 = nn.BatchNorm2d(middle_channle) #同
        self.relu1 = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(middle_channle,in_channle,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(in_channle)
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self,x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x += residual

        return x


class DarkNet(nn.Module):
    def __init__(self,channles,blocks,num_class,need_predict=False):
        super(DarkNet,self).__init__() #继承DarkNet父类的__init__中的参数设置
        self.np = need_predict
        self.channle = 32
        self.conv1 = nn.Conv2d(3,32,kernel_size=3, stride=1, padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.LeakyReLU(0.1)

        # self.blocks = []
        # for i in range(len(blocks)):
        #     self.blocks.append(self._make_layer(channles[i],blocks[i]))
        self.block0 = self._make_layer(channles[0],blocks[0])
        self.block1 = self._make_layer(channles[1],blocks[1])
        self.block2 = self._make_layer(channles[2],blocks[2])
        self.block3 = self._make_layer(channles[3],blocks[3])
        self.block4 = self._make_layer(channles[4],blocks[4])


        if self.np:
            self.avgpool= nn.AdaptiveAvgPool2d((1,1))
            self.fc = nn.Linear(channles[-1][-1],num_class)

        #参数初始化

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self,channles,numbers):
        layers = []
        conv = nn.Conv2d(channles[0], channles[1], kernel_size=3, stride=2, padding=1,bias=False)
        bn = nn.BatchNorm2d(channles[1])
        relu = nn.LeakyReLU(0.1)
        layers.append(('downsample_conv',conv))
        layers.append(('downsample_bn',bn))
        layers.append(('downsample_relu',relu))

        for i in range(0,numbers):
            layers.append((f'ResiduleBlok{i}',ResiduleBlock(channles[1],channles[0])))
        return nn.Sequential(OrderedDict(layers))

    def forward(self,x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.block0(x)
        x = self.block1(x)
        xf1 = self.block2(x)
        xf2 = self.block3(xf1)
        xf3 = self.block4(xf2)

        if self.np:
            x = self.avgpool(xf3)
            x = torch.flatten(x,1)
            x = self.fc(x)
            return x
        else:
            return xf1,xf2,xf3


def darknet53(num_class,need_predict=False):
    channles = [[32, 64], [64, 128], [128, 256], [256, 512], [512, 1024]]
    blocks = [1, 2, 8, 8, 4]
    model = DarkNet(channles, blocks, num_class=num_class, need_predict=need_predict)
    return model


if __name__ == '__main__':
    channles = [[32,64],[64,128],[128,256],[256,512],[512,1024]]
    blocks = [1,2,8,8,4]
    darknet53  = DarkNet(channles,blocks,num_class=20,need_predict=True)


