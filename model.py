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
            nn.Conv2d(self.mid_layer, self.mid_layer//2, kernel_size=self.kernel_size, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(self.mid_layer//2),
            nn.ReLU(inplace=True)
        )

        self.up = nn.ConvTranspose2d(128, 256, 3, stride=2)

        self.conv_head2 = nn.Sequential(
            nn.Conv2d(self.mid_layer//2, self.mid_layer//4, kernel_size=self.kernel_size, padding=1, bias=True),
            nn.BatchNorm2d(self.mid_layer//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_layer//4, self.output_size, kernel_size=self.kernel_size, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(self.output_size),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv_head1(x)
        print(x.shape)
        # torch.Size([1, 64, 128, 128])
        x = self.up(x)
        #x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)     
        # torch.Size([1, 64, 256, 256])
        x = self.conv_head2(x)
        # torch.Size([1, 19, 256, 256])
        return x


def main():

    torch_model = Seg_Head()
    x = torch.randn(1, 384, 128, 128, requires_grad=True)
    torch_out = torch_model(x)
    print(torch_out.shape)
    torch.onnx.export(torch_model, x, "Seg_Head.onnx", export_params=True, opset_version=11,          
                   do_constant_folding=True, input_names = ['input'], output_names = ['output'])

if __name__ == "__main__":
    main()
    