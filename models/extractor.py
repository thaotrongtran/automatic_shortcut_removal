import torchvision.models as models
import torch.nn as nn

class Resnet_FC(nn.Module):
    def __init__(self, out_classes):
        super().__init__()
        #Feature extraction
        res = models.resnet50()
        res.fc = nn.Linear(in_features=2048, out_features=out_classes, bias=True)
        self.res = res
    def forward(self, x):
        x = self.res(x)
        return x