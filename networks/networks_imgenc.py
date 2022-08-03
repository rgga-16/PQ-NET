import torch.nn as nn
from torchvision.models import resnet18


class ImageEncoder(nn.Module):
    def __init__(self, z_dim=512):
        super(ImageEncoder, self).__init__()
        resnet = resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.fc = nn.Sequential(nn.Linear(512, z_dim))

    def forward(self, x):
        feature = self.resnet(x)
        out = self.fc(feature.squeeze())
        return out