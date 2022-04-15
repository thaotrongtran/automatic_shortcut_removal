import torch
import torch.nn as nn

class class_head(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Sequential(
         nn.Flatten(),
         nn.BatchNorm1d(in_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
         nn.Linear(in_features=in_features, out_features=out_features, bias=False))
    def forward(self, x):
        x = self.pooling(x)
        x = self.fc(x)
        return x

class Logistic_Net(nn.Module):
    def __init__(self, pretrained_extractor_path, pretrained_lens_path = None):
        super().__init__()
        self.lens_usage = pretrained_lens_path
        if pretrained_lens_path:
            self.lens = torch.load(pretrained_lens_path)
            for param in self.lens.parameters():
                param.requires_grad = False
        res = torch.load(pretrained_extractor_path)
        self.extractor = nn.Sequential(*list(list(res.children())[0].children())[0:-2])
        self.extractor.eval()
        for param in self.extractor.parameters():
            param.requires_grad = False
        self.head = class_head(in_features=2048, out_features=10)

    def forward(self, x):
        if self.lens_usage:
            x = self.lens(x)
        x = self.extractor(x)
        x = self.head(x)
        return x