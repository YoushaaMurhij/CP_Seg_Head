import torch
import torch.nn as nn
import torch.nn.functional as F

# input tensor [1, 384, 128, 128]

class Seg_Head(nn.Module):
    """Sematntic segmentation head"""

    def __init__(self):
        super(Seg_Head, self).__init__()
        
        self.input_size = 384
        self.mid_layer = 128
        self.output_size = 33
        self.kernel_size = 3

        self.conv_head1 = nn.Sequential(
            nn.Conv2d(self.input_size, self.mid_layer, kernel_size=self.kernel_size, padding=1, bias=True),
            nn.BatchNorm2d(self.mid_layer),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_layer, self.mid_layer//2, kernel_size=self.kernel_size, stride=1, padding=1, bias=True)
        )

        self.conv_head2 = nn.Sequential(
            nn.Conv2d(self.mid_layer//2, self.mid_layer//4, kernel_size=self.kernel_size, padding=1, bias=True),
            nn.BatchNorm2d(self.mid_layer//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_layer//4, self.output_size, kernel_size=self.kernel_size, stride=1, padding=1, bias=True)
        )

    def forward(self, x):
        x = self.conv_head1(x)
        # torch.Size([1, 64, 128, 128])
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)     # Is it necessary!!
        # torch.Size([1, 64, 256, 256])
        x = self.conv_head2(x)
        # torch.Size([1, 19, 256, 256])
        return x
