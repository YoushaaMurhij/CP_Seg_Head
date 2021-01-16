import torch
import onnx
from torch import nn
import torch.nn.functional as F

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


def main():

    torch_model = UNET(384,33)
    x = torch.randn(1, 384, 128, 128, requires_grad=True)
    torch.onnx.export(torch_model, x, "UNET_Head.onnx", export_params=True, opset_version=11,          
                   do_constant_folding=True, input_names = ['input'], output_names = ['output'])
    model = onnx.load("UNET_Head.onnx")
    model_with_shapes = onnx.shape_inference.infer_shapes(model)
    onnx.save(model_with_shapes, "UNET_Head_with_shapes.onnx")

if __name__ == "__main__":
    main()
    