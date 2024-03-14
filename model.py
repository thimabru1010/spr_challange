from torchvision.models import resnet50, resnet34
import torch.nn as nn
import torch
from torchsummary import summary
# from torchinfo import summary

class RegressionModel(nn.Module):
    def __init__(self, in_shape, model_name, **kwargs):
        super(RegressionModel, self).__init__()
        self.in_shape = in_shape
        if model_name == 'resnet50':
            self.model = resnet50(weights=None)
            # self.model = resnet50(weights=weights)
            # self.model.load_state_dict(ResNet50_Weights.DEFAULT)
        elif model_name == 'resnet34':
            self.model = resnet34(weights=None)
            # self.model.load_state_dict(ResNet34_Weights)

        num_channels = 36  # for grayscale images, but it could be any number
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
        
        # print(self.model)
        summary(self.model, tuple(self.in_shape), device='cpu')
        
    def forward(self, x):
        y = self.model(x)
        print(y.shape)
        y = self.fc(y)
        return y
        return self.model(x)
    
if __name__ == '__main__':
    model = RegressionModel(in_shape=(36, 512, 512), model_name='resnet34')
    
    x = torch.randn(1, 36, 512, 512).cpu()
    y = model(x)
    print(y.shape)
    print(y)