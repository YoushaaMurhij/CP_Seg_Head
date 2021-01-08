import torch
import torch.nn as nn
import torch.nn.functional as F

# input tensor [1, 384, 128, 128]

class Seg_Head(nn.Module):

    def __init__(self):
        super(Seg_Head, self).__init__()
         
        self.conv_head1 = nn.Sequential(
            nn.Conv2d(384, 128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=True)
        )

        self.conv_head2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 19, kernel_size=3, stride=1, padding=1, bias=True)
        )

    def forward(self, x):
        x = self.conv_head1(x)
        # torch.Size([1, 64, 128, 128])
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        # torch.Size([1, 64, 256, 256])
        x = self.conv_head2(x)
        # torch.Size([1, 19, 256, 256])
        return x
