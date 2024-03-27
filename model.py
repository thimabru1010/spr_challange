from torchvision.models import resnet50, resnet34, resnet18, efficientnet_b3, efficientnet_b4, swin_v2_b
import torch.nn as nn
import torch
from torchsummary import summary
# from torchinfo import summary

class RegressionModel(nn.Module):
    def __init__(self, in_shape, model_name, aux_clssf=False, input_channels=1, **kwargs):
        super(RegressionModel, self).__init__()
        self.in_shape = in_shape
        if model_name == 'resnet50':
            self.model = resnet50(weights=None)
        elif model_name == 'resnet34':
            self.model = resnet34(weights=None)
        elif model_name == 'resnet18':
            self.model = resnet18(weights=None)
        elif model_name == 'efficientnet-b3':
            self.model = efficientnet_b3(weights=None)
        elif model_name == 'efficientnet-b4':
            self.model = efficientnet_b4(weights=None)
        elif model_name == 'swin':
            self.model = swin_v2_b(weights=None)

        # summary(self.model, tuple(self.in_shape), device='cpu')
        
        num_channels = input_channels  # for grayscale images, but it could be any number
        # Extract the first conv layer's parameters
        num_filters = self.model.conv1.out_channels
        kernel_size = self.model.conv1.kernel_size
        stride = self.model.conv1.stride
        padding = self.model.conv1.padding
        # initialize a new convolutional layer
        conv1 = torch.nn.Conv2d(num_channels, num_filters, kernel_size=kernel_size, stride=stride, padding=padding)
        # Initialize the new conv1 layer's weights by averaging the pretrained weights across the channel dimension
        original_weights = self.model.conv1.weight.data.mean(dim=1, keepdim=True)
        # Expand the averaged weights to the number of input channels of the new dataset
        conv1.weight.data = original_weights.repeat(1, num_channels, 1, 1)
        self.model.conv1 = conv1
        
        # self.model.fc = nn.Linear(512, 1)
        # Remove the last layer
        self.model.fc = nn.Identity()
        self.fc = nn.Linear(512, 1)
        
        self.fc2 = nn.Identity()
        if aux_clssf:
            self.fc2 = nn.Linear(512, 3)
        
        
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
    
    x = torch.randn(1, 1, 512, 512).cpu()
    y = model(x)
    print(y.shape)
    print(y)
    
    
    
