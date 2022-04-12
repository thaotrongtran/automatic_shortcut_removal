import torch
import torch.nn as nn

class Logistic_Net(nn.Module):
    def __init__(self, pretrained_extractor_path, pretrained_lens_path = None):
        super().__init__()
        self.lens_usage = pretrained_lens_path
        if pretrained_lens_path:
            self.lens = torch.load(pretrained_lens_path)
            for param in self.lens.parameters():
                param.requires_grad = False
        self.extractor = torch.load(pretrained_extractor_path)
        self.extractor.eval()
        for param in self.extractor.parameters():
            param.requires_grad = False
        num_channels = 2048
        num_feats = 1000
        num_classes = 10
        self.extractor.res.fc = nn.Sequential(
            nn.Linear(num_channels, num_feats, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_feats, num_feats, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_feats, num_classes)
        )

    def forward(self, x):
        if self.lens_usage:
            x = self.lens(x)
        x = self.extractor(x)
        return x