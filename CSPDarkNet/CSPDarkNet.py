import math
import torch
import torch.nn as nn
import torch.nn.functional as F


#Mish激活函数
class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

#Conv + Bn + Mish
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = Mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class ResBlock(nn.Module):
    def __init__(self,outChannel):
        super(ResBlock, self).__init__()
        if outChannel==64:
            self.resBlock = nn.Sequential(BasicConv(outChannel,32,kernel_size=1,stride=1),
                                          BasicConv(32,outChannel,kernel_size=3,stride=1),
                                        )
        else:
            resChannel = outChannel//2
            self.resBlock = nn.Sequential(BasicConv(resChannel,resChannel,kernel_size=1,stride=1),
                                          BasicConv(resChannel,resChannel,kernel_size=3,stride=1)
                                          )

    def forward(self,x):
        x = x+self.resBlock(x)
        return x

class LayerNet(nn.Module):
    def __init__(self,blockNum,outChannel):
        super(LayerNet, self).__init__()
        self.downSample = BasicConv(outChannel//2,outChannel,kernel_size=3,stride=2)
        if blockNum==1:
            self.conv0 = BasicConv(outChannel,outChannel,kernel_size=1,stride=1)
            self.conv1 = BasicConv(outChannel,outChannel,kernel_size=1,stride=1)
            self.resBlock = ResBlock(outChannel)
            self.conv2 = BasicConv(outChannel,outChannel,kernel_size=1,stride=1)
            self.conv3 = BasicConv(outChannel*2,outChannel,kernel_size=1,stride=1)
        else:
            self.conv0 = BasicConv(outChannel, outChannel//2, kernel_size=1, stride=1)
            self.conv1 = BasicConv(outChannel, outChannel//2, kernel_size=1, stride=1)
            self.resBlock = nn.Sequential(*[ResBlock(outChannel) for i in range(blockNum)])
            self.conv2 = BasicConv(outChannel//2, outChannel//2, kernel_size=1, stride=1)
            self.conv3 = BasicConv(outChannel,outChannel,kernel_size=1,stride=1)


    def forward(self,x):
        x = self.downSample(x)
        x0 = self.conv0(x)
        x1 = self.conv1(x)
        x1 = self.resBlock(x1)
        x1 = self.conv2(x1)

        x = torch.cat([x0,x1],dim=1)
        x = self.conv3(x)

        return x


class CSPDarkNet(nn.Module):
    def __init__(self,blockNums,outChannels,numClass=0,classify=False):
        super(CSPDarkNet, self).__init__()
        self.basicLayer1 = BasicConv(3,32,kernel_size=3,stride=1)

        # self.stages = nn.ModuleList([
        #     LayerNet(blockNums[0], outChannels[0]),
        #     LayerNet(blockNums[1], outChannels[1]),
        #     LayerNet(blockNums[2], outChannels[2]),
        #     LayerNet(blockNums[3], outChannels[3]),
        #     LayerNet(blockNums[4], outChannels[4])
        # ])

        self.res_layer0 = LayerNet(blockNums[0], outChannels[0]) #150
        self.res_layer1 = LayerNet(blockNums[1], outChannels[1]) #75
        self.res_layer2 = LayerNet(blockNums[2], outChannels[2]) #38 256
        self.res_layer3 = LayerNet(blockNums[3], outChannels[3]) #19 512
        self.res_layer4 = LayerNet(blockNums[4], outChannels[4]) #10 1024


        self.classify = classify
        if classify:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(outChannels[-1], numClass)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self,x):
        x = self.basicLayer1(x)

        # x =self.stages[0](x)
        # x = self.stages[1](x)
        # out3 = self.stages[2](x)
        # out4 = self.stages[3](out3)
        # out5 = self.stages[4](out4)

        x = self.res_layer0(x)
        x = self.res_layer1(x)
        out3 = self.res_layer2(x)
        out4 = self.res_layer3(out3)
        out5 = self.res_layer4(out4)

        if self.classify:
            x = self.avgpool(out5)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return  x

        return out3, out4, out5


def CSPDarkNet53(pretrainedWeight=False):
    blockNums = [1,2,8,8,4]
    outChannles = [64,128,256,512,1024]
    net = CSPDarkNet(blockNums,outChannles)
    if pretrainedWeight:
        if isinstance(pretrainedWeight,str):
            net.load_state_dict(torch.load(pretrainedWeight))
        else:
            raise Exception(f'net pretrained weights path is false:{pretrainedWeight}')

    return net


if __name__ == '__main__':
    net = CSPDarkNet53()
    print(list(net.children())[3])
