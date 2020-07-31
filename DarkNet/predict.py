import torch
from DarkNet import darknet53
from torchvision import transforms
import json
from PIL import Image
import matplotlib.pyplot as plt


#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#图片地址
image_root = 'D:/DataSet/class_voc/test/timg.jpg'

data_transform = transforms.Compose(
    [transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

img = Image.open(image_root)
img = data_transform(img)
img = torch.unsqueeze(img,dim=0)

#类别文件
class_json = open('D:\DataSet\class_voc/class_indices_voc.json','r')
try:
    class_indict = json.load(class_json)
except Exception as e:
    print(e)
    exit(-1)

#创建网络
net = darknet53(num_class=20,need_predict=True)
model_weight = './Darknet53.pth'
net.load_state_dict(torch.load(model_weight))
net.eval()
#预测
with torch.no_grad():
    output = torch.squeeze(net(img))
    predict = torch.softmax(output,dim=0)
    predict_class = torch.argmax(predict).numpy()
print(f'预测类别：{class_indict[str(predict_class)]},概率：{predict[predict_class].numpy()}')

