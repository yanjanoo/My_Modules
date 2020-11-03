import torch.nn as nn
import torch

#上采样
class UpSample(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(UpSample, self).__init__()
        self.up_sample = nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size=1),
                                       nn.Upsample(scale_factor=2,mode='nearest'))

    def forward(self,x):
        x = self.up_sample(x)
        return x

#vgg16特征提取层
class BasicLayer(nn.Module):
    def __init__(self):
        super(BasicLayer, self).__init__()
        self.conv11 = nn.Conv2d(3,64,kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(64,64,kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)


        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)


        self.conv41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)


        self.conv51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  #1/2

        x = self.conv21(x)
        x = self.conv22(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x) #1/4

        x = self.conv31(x)
        x = self.conv32(x)
        x = self.conv33(x)
        x = self.bn3(x)
        x = self.relu(x)
        out1 = self.maxpool(x)  #1/8

        x = self.conv41(out1)
        x = self.conv42(x)
        x = self.conv43(x)
        x = self.bn4(x)
        x = self.relu(x)
        out2 = self.maxpool(x)   #1/16

        x = self.conv51(out2)
        x = self.conv52(x)
        x = self.conv53(x)
        x = self.bn5(x)
        x = self.relu(x)
        out3 = self.maxpool(x)  # 1/32

        return out1,out2,out3


class VggPro(nn.Module):
    def __init__(self,num_classes=20,init_weights=False):
        super(VggPro, self).__init__()
        self.basic_layer = BasicLayer()
        self.up_sample = UpSample(512,512)
        self.down_sample = nn.Conv2d(256,512,kernel_size=3,stride=2,padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512 * 7 * 7, 2048),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Linear(2048, num_classes)
        )
        if init_weights:
            self._initialize_weights()


    def forward(self,x):
        p1,p2,p3 = self.basic_layer(x)

        p3 = self.up_sample(p3)
        p1 = self.down_sample(p1)

        p2 = p2+p3+p1

        x = self.max_pool(p2)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
