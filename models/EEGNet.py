import torch.nn as nn
import math
import torch


# class EEGNet(nn.Module):
#     def __init__(self, num_channels=2, num_classes=2):
#         super(EEGNet, self).__init__()
#         self.firstconv = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False),
#             nn.BatchNorm2d(16, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
#         )

#         self.depthwiseConv = nn.Sequential(
#             nn.Conv2d(16, 32, kernel_size=(num_channels, 1), stride=(1, 1), groups=16, bias=False),
#             nn.BatchNorm2d(32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
#             # nn.ELU(alpha=0.001),
#             nn.ELU(),
#             nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
#             nn.Dropout(p=0.35)
#         )

#         self.separableConv = nn.Sequential(
#             nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
#             nn.BatchNorm2d(32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
#             # nn.ELU(alpha=0.001),
#             nn.ELU(),
#             nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
#             nn.Dropout(p=0.35)
#         )

#         self.classify = nn.Sequential(
#             nn.Linear(736, num_classes),
#             # nn.Linear(736, 128),
#             # nn.ReLU(), 
#             # nn.Dropout(p=0.25),
#             # nn.Linear(128, num_classes), 
#         )

#     def forward(self, x):
#         x = self.firstconv(x)
#         x = self.depthwiseConv(x)
#         x = self.separableConv(x)
#         x = x.view(x.size(0), -1)  # Flatten
#         x = self.classify(x)
#         return x


class EEGNet(nn.Module):
    def __init__(self, num_channels=2, num_classes=2):
        super(EEGNet, self).__init__()
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 31), stride=(1, 1), padding=(0, 15), bias=False),
            nn.BatchNorm2d(16, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        )

        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(num_channels, 15), stride=(1, 1), groups=16, bias=False),
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            # nn.ReLU(0.1),
            nn.ELU(alpha=0.001),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            nn.Dropout(p=0.35)
        )

        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 11), stride=(1, 1), padding=(0, 5), bias=False),
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            # nn.ReLU(0.1),
            nn.ELU(alpha=0.001),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(p=0.35)
        )

        self.classify = nn.Sequential(
            nn.Linear(736, num_classes),
        )

    def forward(self, x):
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = x.view(x.size(0), -1)
        x = self.classify(x)
        return x




# (Optional) implement DeepConvNet model
class DeepConvNet(nn.Module):
    def __init__(self):
        super(DeepConvNet, self).__init__()
        pass

    def forward(self, x):
        pass

print(EEGNet())