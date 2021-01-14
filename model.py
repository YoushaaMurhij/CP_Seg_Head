import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx

# input tensor [1, 384, 128, 128]

class Seg_Head(nn.Module):
    """Sematntic segmentation head"""

    def __init__(self):
        super(Seg_Head, self).__init__()
        
        self.input_size = 384
        self.mid_layer = 128
        self.output_size = 33
        self.kernel_size = 1

        self.conv_head1 = nn.Sequential(
            nn.Conv2d(self.input_size, self.mid_layer, kernel_size=self.kernel_size, padding=0, bias=True),
            nn.BatchNorm2d(self.mid_layer),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_layer, self.mid_layer//2, kernel_size=self.kernel_size, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(self.mid_layer//2),
            nn.ReLU(inplace=True)
        )

        #self.up = nn.ConvTranspose2d(self.mid_layer//2, self.mid_layer//4, 2, stride=2, padding=0)

        self.conv_head2 = nn.Sequential(
            nn.Conv2d(self.mid_layer//2, self.mid_layer//4, kernel_size=self.kernel_size, padding=0, bias=True),
            nn.BatchNorm2d(self.mid_layer//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_layer//4, self.output_size, kernel_size=self.kernel_size, stride=1, padding=0, bias=True),
        )

    def forward(self, x):
        x = self.conv_head1(x)
        #x = self.up(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)     
        x = self.conv_head2(x)
        #print(x.shape)
        return x


def main():

    torch_model = Seg_Head()
    x = torch.randn(1, 384, 128, 128, requires_grad=True)
    torch.onnx.export(torch_model, x, "Seg_Head.onnx", export_params=True, opset_version=11,          
                   do_constant_folding=True, input_names = ['input'], output_names = ['output'])
    model = onnx.load("Seg_Head.onnx")
    model_with_shapes = onnx.shape_inference.infer_shapes(model)
    onnx.save(model_with_shapes, "Seg_Head_with_shapes.onnx")

if __name__ == "__main__":
    main()
    