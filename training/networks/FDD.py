import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward


class FDD(nn.Module): 
    def __init__(self, in_ch, out_ch):
        super(FDD, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
                                    nn.Conv2d(in_ch*3, in_ch, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(in_ch),
                                    nn.ReLU(inplace=True),
                                    )
        self.outconv_bn_relu_L = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.outconv_bn_relu_H = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:,:,0,::]
        y_LH = yH[0][:,:,1,::]
        y_HH = yH[0][:,:,2,::]
        yH = torch.cat([y_HL, y_LH, y_HH], dim=1)
        yH = self.conv_bn_relu(yH)
        yL = self.outconv_bn_relu_L(yL)
        yH = self.outconv_bn_relu_H(yH)
        return yL,yH


class FDDown(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(FDDown, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
                                    nn.Conv2d(in_ch*3, in_ch, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(in_ch),
                                    nn.ReLU(inplace=True),
                                    )
        self.outconv_bn_relu_L = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.outconv_bn_relu_H = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:,:,0,::]
        y_LH = yH[0][:,:,1,::]
        y_HH = yH[0][:,:,2,::]
        yH = torch.cat([y_HL, y_LH, y_HH], dim=1)
        yH = self.conv_bn_relu(yH)
        yL = self.outconv_bn_relu_L(yL)
        yH = self.outconv_bn_relu_H(yH)
        return yL + yH 

if __name__ == "__main__":
    input = torch.randn(1,32, 64, 64)
    FDD =  FDD(32,32)
    output_L,output_H = FDD(input) 
    print(f"input  shape: {input.shape}")
    print(f"output_L shape: {output_L.shape}")
    print(f"output_H shape: {output_H.shape}")

    print('-----------')
    FDDown = FDDown(32,64)
    output = FDDown(input)
    print(f"input  shape: {input.shape}")
    print(f"output shape: {output.shape}")