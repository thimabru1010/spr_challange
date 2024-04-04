from torchvision.models import resnet50, resnet34, resnet18, efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, swin_v2_t, swin_v2_s, densenet121
import torch.nn as nn
import torch
from torchsummary import summary
# from torchinfo import summary

class RegressionModel(nn.Module):
    def __init__(self, in_shape, model_name, aux_clssf=False, input_channels=1, **kwargs):
        super(RegressionModel, self).__init__()
        self.in_shape = in_shape
        print(f'Using model: {model_name}')
        if 'resnet' in model_name:
            if model_name == 'resnet50':
                self.model = resnet50(weights=None)
                output_size = 2048
            elif model_name == 'resnet34':
                self.model = resnet34(weights=None)
                output_size = 512
            elif model_name == 'resnet18':
                self.model = resnet18(weights=None)
                output_size = 512
            self.model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2),\
                padding=(3, 3), bias=False)
            self.model.fc = nn.Identity()
        elif model_name == 'densenet121':
            self.model = densenet121(weights=None)
            self.model.features[0] = nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.model.classifier[0] = nn.Identity()
            output_size = 1024
        elif model_name == 'efficientnet_b0':
            self.model = efficientnet_b0(weights=None)
            self.model.features[0] = nn.Conv2d(input_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.model.classifier[1] = nn.Identity()
            output_size = 1280
        elif model_name == 'efficientnet_b1':
            self.model = efficientnet_b1(weights=None)
            self.model.features[0] = nn.Conv2d(input_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.model.classifier[1] = nn.Identity()
            output_size = 1280
        elif model_name == 'efficientnet_b2':
            self.model = efficientnet_b2(weights=None)
            self.model.features[0] = nn.Conv2d(input_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.model.classifier[1] = nn.Identity()
            output_size = 1408
        elif model_name == 'efficientnet_b3':
            self.model = efficientnet_b3(weights=None)
            self.model.features[0] = nn.Conv2d(input_channels, 40, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.model.classifier[1] = nn.Identity()
            output_size = 1536
        elif model_name == 'efficientnet_b4':
            self.model = efficientnet_b4(weights=None)
            self.model.features[0] = nn.Conv2d(input_channels, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.model.classifier[1] = nn.Identity()
            output_size = 1792
        elif model_name == 'swin_s':
            self.model = swin_v2_s(weights=None)
            self.model.features[0][0] = nn.Conv2d(input_channels, 96, kernel_size=(4, 4), stride=(4, 4), padding=(1, 1), bias=False)
            self.model.head = nn.Identity()
            output_size = 768
        elif model_name == 'swin_t':
            self.model = swin_v2_t(weights=None)
            self.model.features[0][0] = nn.Conv2d(input_channels, 96, kernel_size=(4, 4), stride=(4, 4), padding=(1, 1), bias=False)
            self.model.head = nn.Identity()
            output_size = 768
        print(self.model)
        
        self.fc = nn.Linear(output_size, 1)
        
        self.fc2 = nn.Identity()
        if aux_clssf:
            self.fc2 = nn.Linear(output_size, 3)
        
        # print(self.model)
        # summary(self.model, tuple(self.in_shape), device='cpu')
        
    def forward(self, x):
        y = self.model(x)
        # print(y.shape)
        y_reg = self.fc(y)
        y_clssf = self.fc2(y)
        return y_reg, y_clssf
 

class RegressionModel2(nn.Module):
	#  Determine what layers and their order in CNN object 
    def __init__(self, in_shape, **kwargs):
        super(RegressionModel2, self).__init__()
        
        self.in_shape = in_shape
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer6 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1152, 512),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(512, 1))
    
    # Progresses data across layers    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out      
        



 
if __name__ == '__main__':
    model = RegressionModel(in_shape=(1, 512, 512), model_name='resnet18')
    # model = RegressionModel2(in_shape=(1, 512, 512))
    
    x = torch.randn(1, 9, 512, 512).cpu()
    y = model(x)
    print(y.shape)
    print(y)
    
    
    
