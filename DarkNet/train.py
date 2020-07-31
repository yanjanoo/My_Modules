import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from DarkNet import darknet53

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
save_path = './Darknet53.pth'

#数据准备
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "val": transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
#图片地址
image_root = 'D:/DataSet/class_voc/'
#数据预处理
train_data = datasets.ImageFolder(root = image_root+'train',transform=data_transform["train"])
val_data = datasets.ImageFolder(root = image_root+'val',transform=data_transform["val"])
#数据按批次加载
train_loader = torch.utils.data.DataLoader(train_data,batch_size=32,shuffle=True,num_workers=0)
val_loader = torch.utils.data.DataLoader(val_data,batch_size=32,shuffle=False,num_workers=0)

train_num = len(train_data)
val_num = len(val_data)



#模型创建
net = darknet53(num_class=20,need_predict=True)
#迁移学习
model_pre_weight = './Darknet53.pth'
net.load_state_dict(torch.load(model_pre_weight),strict=False)
net.to(device)
loss_function = nn.CrossEntropyLoss() #损失函数
optimizer = optim.Adam(net.parameters(),lr=0.0001)  #优化方法


#网络训练
best_acc = 0.0
for epoch in range(30):
    #训练
    net.train()
    running_loss = 0.0
    for i,data in enumerate(train_loader,start=0):
        images,labels = data
        optimizer.zero_grad()        #梯度归零
        outputs = net(images.to(device))
        loss = loss_function(outputs,labels.to(device))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        rate = (i + 1) / len(train_loader)
        a = "*" * int(rate * 100)
        b = "." * (100-int(rate* 100))
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
    print()

    #验证
    net.eval()
    acc = 0.0
    with torch.no_grad():
        for val_data in val_loader:
            val_images,val_labels = val_data
            optimizer.zero_grad()
            outputs = net(val_images.to(device))
            predict_y = torch.max(outputs,dim=1)[1] #one-hot,返回最大值索引
            acc += (predict_y == val_labels.to(device)).sum().item()  #batchs_size的sum()
        val_acc = acc/val_num
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(),save_path)
        print(f'epoch:{epoch+1},train_loss:{running_loss/i:.3f},test_accuracy:{val_acc:.3f}')

print('Finished Training')