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
            self.model = resnet50()
            # self.model = resnet50(weights=weights)
            # self.model.load_state_dict(ResNet50_Weights.DEFAULT)
        elif model_name == 'resnet34':
            self.model = resnet34()
            # self.model.load_state_dict(ResNet34_Weights)

        self.model.fc = nn.Linear(512, 1)
        
        # print(self.model)
        summary(self.model, tuple(self.in_shape), device='cpu')
        
    def forward(self, x):
        return self.model(x)
    
if __name__ == '__main__':
    model = RegressionModel(in_shape=(3, 64, 64), model_name='resnet34')
    
    x = torch.randn(1, 3, 64, 64).cpu()
    y = model(x)
    print(y.shape)
    print(y)