import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import torchvision

# input tensor [1, 384, 128, 128]

class Seg_Head(nn.Module):
    """Sematntic segmentation head"""

    def __init__(self):
        super(Seg_Head, self).__init__()
        
        self.input_size = 384
        self.mid_layer = 256
        self.output_size = 26
        self.kernel_size = 1
        self.dropout = 0.1

        self.conv_head1 = nn.Sequential(
            nn.Conv2d(self.input_size, self.mid_layer, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Dropout(self.dropout),
            nn.BatchNorm2d(self.mid_layer),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_layer, self.mid_layer//2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Dropout(self.dropout),
            nn.BatchNorm2d(self.mid_layer//2),
            nn.ReLU(inplace=True)
        )
        self.conv_head2 = nn.Sequential(
            nn.Conv2d(self.mid_layer//2, self.mid_layer//4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Dropout(self.dropout),
            nn.BatchNorm2d(self.mid_layer//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_layer//4, self.mid_layer//8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Dropout(self.dropout),
            nn.BatchNorm2d(self.mid_layer//8),
            nn.ReLU(inplace=True)
        )

        #self.up = nn.ConvTranspose2d(self.mid_layer//2, self.mid_layer//4, 2, stride=2, padding=0)
        self.conv_head3 = nn.Sequential(
            nn.Conv2d(self.mid_layer//8, self.output_size, kernel_size=self.kernel_size, stride=1, padding=0, bias=True),
        )

    def forward(self, x):
        x = self.conv_head1(x)
        x = self.conv_head2(x)
        x = self.conv_head3(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return x

class UNET(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = self.contract_block(in_channels, 32, 7, 3)
        self.conv2 = self.contract_block(32, 64, 3, 1)
        self.conv3 = self.contract_block(64, 128, 3, 1)

        self.upconv3 = self.expand_block(128, 64, 3, 1)
        self.upconv2 = self.expand_block(64*2, 32, 3, 1)
        self.upconv1 = self.expand_block(32*2, 33, 3, 1)

    def __call__(self, x):

        # downsampling part
        #print(x.shape)
        conv1 = self.conv1(x)
        #print(conv1.shape)
        conv2 = self.conv2(conv1)
        #print(conv2.shape)
        conv3 = self.conv3(conv2)
        #print(conv3.shape)

        upconv3 = self.upconv3(conv3)
        #print(upconv3.shape)

        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        #print(upconv2.shape)
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))
        #print(upconv1.shape)
        y = F.interpolate(upconv1, scale_factor=2, mode='bilinear', align_corners=True)
        #print(y.shape)

        return y

    def contract_block(self, in_channels, out_channels, kernel_size, padding):

        contract = nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                                 )

        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):

        expand = nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
                            torch.nn.BatchNorm2d(out_channels),
                            torch.nn.ReLU(),
                            torch.nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
                            torch.nn.BatchNorm2d(out_channels),
                            torch.nn.ReLU(),
                            torch.nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1) 
                            )
        return expand

class New_Head(nn.Module):
    """Sematntic segmentation head"""

    def __init__(self, model):
        super(New_Head, self).__init__()
        
        self.input_size = 384
        self.mid_layer = 128
        self.output_size = 3
        self.kernel_size = 1
        self.conv_head1 = nn.Sequential(
            nn.Conv2d(self.input_size, self.mid_layer, kernel_size=self.kernel_size, padding=0, bias=True),
            nn.BatchNorm2d(self.mid_layer),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_layer, self.mid_layer//2, kernel_size=self.kernel_size, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(self.mid_layer//2),
            nn.ReLU(inplace=True)
        )
        self.conv_head2 = nn.Sequential(
            nn.Conv2d(self.mid_layer//2, self.mid_layer//4, kernel_size=self.kernel_size, padding=0, bias=True),
            nn.BatchNorm2d(self.mid_layer//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_layer//4, self.output_size, kernel_size=self.kernel_size, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(self.output_size),
            nn.ReLU(inplace=True)
        )
        self.model = model

    def forward(self, x):
        x = self.conv_head1(x)
        x = self.conv_head2(x)
        x = self.model(x)['out']
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return x

def get_model():
    resnet = torchvision.models.segmentation.fcn_resnet50(pretrained=False, progress=True, num_classes=33, aux_loss=None)
    model = New_Head(resnet)
    return model

def main():
    torch_model = Seg_Head()
    x = torch.randn(1, 384, 128, 128, requires_grad=True)
    y = torch_model(x)
    torch.onnx.export(torch_model, x, "New_Head.onnx", export_params=True, opset_version=11,          
                   do_constant_folding=False, input_names = ['input'], output_names = ['output'])
    model = onnx.load("New_Head.onnx")
    model_with_shapes = onnx.shape_inference.infer_shapes(model)
    onnx.save(model_with_shapes, "New_Head_with_shapes.onnx")

if __name__ == "__main__":
    main()
    