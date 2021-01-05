import torch
import torch.nn as nn
import torch.nn.functional as F

from det3d.models.utils import Sequential
# input tensor [1, 384, 128, 128]
# output

class Seg_Head(nn.Module):

    def __init__(self, init_bias=-2.19):
        super(Seg_Head, self).__init__()
         
        self.conv_head = Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=True)
        )

        # Focal loss paper points out that it is important to initialize the bias 
        self.conv_head[-1].bias.data.fill_(init_bias)


    def forward(self, x):
        x = self.conv_head(x)
        up_level1 = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
       
       
        return x



net = Net()
print(net)