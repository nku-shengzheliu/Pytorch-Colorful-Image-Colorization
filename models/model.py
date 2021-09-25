import torch
import torch.nn as nn
from torchvision import models

def weights_init(model):
    if type(model) in [nn.Conv2d, nn.Linear]:
        nn.init.xavier_normal_(model.weight.data)
        nn.init.constant_(model.bias.data, 0.1)

class Color_model(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(Color_model, self).__init__()
        
        model1=[nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True),]
        model1+=[nn.ReLU(True),]
        model1+=[nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),]
        model1+=[nn.ReLU(True),]
        model1+=[norm_layer(64),]

        model2=[nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),]
        model2+=[nn.ReLU(True),]
        model2+=[nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),]
        model2+=[nn.ReLU(True),]
        model2+=[norm_layer(128),]

        model3=[nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model3+=[nn.ReLU(True),]
        model3+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model3+=[nn.ReLU(True),]
        model3+=[nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),]
        model3+=[nn.ReLU(True),]
        model3+=[norm_layer(256),]

        model4=[nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[nn.ReLU(True),]
        model4+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[nn.ReLU(True),]
        model4+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[nn.ReLU(True),]
        model4+=[norm_layer(512),]

        model5=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.ReLU(True),]
        model5+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.ReLU(True),]
        model5+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.ReLU(True),]
        model5+=[norm_layer(512),]

        model6=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.ReLU(True),]
        model6+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.ReLU(True),]
        model6+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.ReLU(True),]
        model6+=[norm_layer(512),]

        model7=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.ReLU(True),]
        model7+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.ReLU(True),]
        model7+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.ReLU(True),]
        model7+=[norm_layer(512),]

        model8=[nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True),]
        model8+=[nn.ReLU(True),]
        model8+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model8+=[nn.ReLU(True),]
        model8+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model8+=[nn.ReLU(True),]

        model8+=[nn.Conv2d(256, 313, kernel_size=1, stride=1, padding=0, bias=True),]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8 = nn.Sequential(*model8)
        self.apply(weights_init)

    def forward(self, gray_image):
        conv1_2 = self.model1(gray_image)  # [bs, 64, 112, 112]
        conv2_2 = self.model2(conv1_2)  # [bs, 128, 56, 56]
        conv3_3 = self.model3(conv2_2)  # [bs, 256, 28, 28]
        conv4_3 = self.model4(conv3_3)  # [bs, 512, 28, 28]
        conv5_3 = self.model5(conv4_3)  # [bs, 512, 28, 28]
        conv6_3 = self.model6(conv5_3)  # [bs, 512, 28, 28]
        conv7_3 = self.model7(conv6_3)  # [bs, 512, 28, 28]
        features = self.model8(conv7_3)  # [bs, 313, 56, 56]
        features = features/0.38  # 0.38 is Softmax temperature T. Paper Eq.(5)
        return features

def load_weights(model, load_path):
    dict_trained = torch.load(load_path)
    # new_list = list(model.state_dict().keys())
    # trained_list = list(dict_trained.keys())
    dict_new = model.state_dict().copy()
    for key in dict_new.keys():
        if key in dict_trained.keys():
            dict_new[key] = dict_trained[key]
        else:
            print(key)
    model.load_state_dict(dict_new)
    del dict_new
    del dict_trained
    torch.cuda.empty_cache()
    return model

if __name__=="__main__":
    x = torch.randn([40, 1, 224, 224])*100
    model = Color_model()
    model_path = 'saved_models/best.pth'
    model = load_weights(model, model_path)
    print('load success')
    import onnx
    # 导出onnx模型文件
    dummy_input = torch.randn([1, 1, 224, 224])
    torch.onnx.export(model, dummy_input, "saved_models/best.onnx",verbose=True)
    print("Dump onnx")   
    # x2paddle --framework=onnx --model=saved_models/best.onnx --save_dir='saved_models/pd_model'
    # output = model(x)
    # print(output.shape)