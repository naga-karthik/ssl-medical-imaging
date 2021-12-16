from math import log
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


""" Parts of the U-Net model 
    Taken from: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
"""
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


""" Full assembly of the parts to form the complete network 
    Adapted from: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
"""
class UNet(nn.Module):
    def __init__(self, n_channels, init_filters, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.init_filters = init_filters
        self.n_classes = n_classes
        self.bilinear = bilinear

        # self.inc = DoubleConv(n_channels, init_filters)
        # self.down1 = Down(init_filters, init_filters*2)
        # self.down2 = Down(init_filters*2, init_filters*4)
        # self.down3 = Down(init_filters*4, init_filters*8)
        # factor = 2 if bilinear else 1
        # self.down4 = Down(init_filters*8, init_filters*16 // factor)
        # self.up1 = Up(init_filters*16, init_filters*8 // factor, bilinear)
        # self.up2 = Up(init_filters*8, init_filters*4 // factor, bilinear)
        # self.up3 = Up(init_filters*4, init_filters*2 // factor, bilinear)
        # self.up4 = Up(init_filters*2, init_filters, bilinear)
        # self.outc = OutConv(init_filters, n_classes)
        self.inc = DoubleConv(n_channels, init_filters)
        self.down1 = Down(init_filters, init_filters*2)
        self.down2 = Down(init_filters*2, init_filters*4)
        self.down3 = Down(init_filters*4, init_filters*8)
        self.down4 = Down(init_filters*8, init_filters*16)
        factor = 2 if bilinear else 1
        self.down5 = Down(init_filters*16, init_filters*32 // factor)
        self.up1 = Up(init_filters*32, init_filters*16 // factor, bilinear)
        self.up2 = Up(init_filters*16, init_filters*8 // factor, bilinear)
        self.up3 = Up(init_filters*8, init_filters*4 // factor, bilinear)
        self.up4 = Up(init_filters*4, init_filters*2 // factor, bilinear)
        self.up5 = Up(init_filters*2, init_filters, bilinear)
        self.outc = OutConv(init_filters, n_classes)

    def forward(self, x):
        # x1 = self.inc(x)
        # x2 = self.down1(x1)
        # x3 = self.down2(x2)
        # x4 = self.down3(x3)
        # x5 = self.down4(x4)
        # x = self.up1(x5, x4)
        # x = self.up2(x, x3)
        # x = self.up3(x, x2)
        # x = self.up4(x, x1)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        logits = self.outc(x)
        out_final = F.softmax(logits, dim=1)
        return logits, out_final


# #############################################################################
# ---------------------------- UNet Encoder Model -----------------------------
# #############################################################################
class UNetEncoder(nn.Module):
    def __init__(self, n_channels, init_filters, bilinear=True):
        super(UNetEncoder, self).__init__()
        self.n_channels = n_channels
        self.init_filters = init_filters
        self.bilinear = bilinear

        # self.inc = DoubleConv(n_channels, init_filters)
        # self.down1 = Down(init_filters, init_filters*2)
        # self.down2 = Down(init_filters*2, init_filters*4)
        # self.down3 = Down(init_filters*4, init_filters*8)
        # factor = 2 if bilinear else 1
        # self.down4 = Down(init_filters*8, init_filters*16 // factor)
        self.inc = DoubleConv(n_channels, init_filters)
        self.down1 = Down(init_filters, init_filters*2)
        self.down2 = Down(init_filters*2, init_filters*4)
        self.down3 = Down(init_filters*4, init_filters*8)
        self.down4 = Down(init_filters*8, init_filters*16)
        factor = 2 if bilinear else 1
        self.down5 = Down(init_filters*16, init_filters*32 // factor)

    def forward(self, x):
        context_features = []
        x1 = self.inc(x)
        context_features.append(x1)
        x2 = self.down1(x1)
        context_features.append(x2)
        x3 = self.down2(x2)
        context_features.append(x3)
        x4 = self.down3(x3)
        context_features.append(x4)
        x5 = self.down4(x4)
        # return x5, context_features
        context_features.append(x5)
        x6 = self.down5(x5)
        return x6, context_features

# #############################################################################
# ---------------------------- UNet Decoder Model -----------------------------
# #############################################################################
class UNetDecoder(nn.Module):
    def __init__(self, init_filters, n_classes, bilinear=True):
        super(UNetDecoder, self).__init__()
        self.init_filters = init_filters
        self.n_classes = n_classes
        self.bilinear = bilinear

        # factor = 2 if bilinear else 1
        # self.up1 = Up(init_filters*16, init_filters*8 // factor, bilinear)
        # self.up2 = Up(init_filters*8, init_filters*4 // factor, bilinear)
        # self.up3 = Up(init_filters*4, init_filters*2 // factor, bilinear)
        # self.up4 = Up(init_filters*2, init_filters, bilinear)
        # self.outc = OutConv(init_filters, n_classes)
        factor = 2 if bilinear else 1
        self.up1 = Up(init_filters*32, init_filters*16 // factor, bilinear)
        self.up2 = Up(init_filters*16, init_filters*8 // factor, bilinear)
        self.up3 = Up(init_filters*8, init_filters*4 // factor, bilinear)
        self.up4 = Up(init_filters*4, init_filters*2 // factor, bilinear)
        self.up5 = Up(init_filters*2, init_filters, bilinear)
        self.outc = OutConv(init_filters, n_classes)
    
    def forward(self, enc_out, context_features):
        # x1, x2, x3, x4 = context_features   # getting these from the encoder     
        # x = self.up1(enc_out, x4)   # enc_out is x5 in the full UNet model
        # x = self.up2(x, x3)
        # x = self.up3(x, x2)
        # x = self.up4(x, x1)
        x1, x2, x3, x4, x5 = context_features   # getting these from the encoder     
        x = self.up1(enc_out, x5)   # enc_out is x6 in the full UNet model
        x = self.up2(x, x4)  
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)

        logits = self.outc(x)
        out_final = F.softmax(logits, dim=1)
        return logits, out_final


# #############################################################################
# ---------------------------- G1 Projector Head -----------------------------
# #############################################################################
class ProjectorHead(nn.Module):
    def __init__(self, encoder_init_filters, out_dim):
        super(ProjectorHead, self).__init__()
        self.out_dim = out_dim
        # self.projector_g1 = nn.Sequential(
        #     nn.Linear(((encoder_init_filters*16)//2)*(12*12), 3200),    # final enc output is 12x12 (ie downsampled to 12x12), which is then multiplied by 128 filters for flattening
        #     nn.ReLU(),
        #     nn.Linear(3200, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, out_dim)
        # )
        self.projector_g1 = nn.Sequential(
            nn.Linear(((encoder_init_filters*32)//2)*(6*6), 1024),    # final enc output is 12x12 (ie downsampled to 12x12), which is then multiplied by 128 filters for flattening
            nn.ReLU(),
            nn.Linear(1024, out_dim)
        )
    def forward(self, encoder_out):
        out = torch.flatten(encoder_out, 1)
        projector_out = self.projector_g1(out)
        
        return projector_out

if __name__ == "__main__":
    input = torch.randn(8, 1, 192, 192)
    # model = ModifiedResnet(downloaded_net, num_classes=4)
    # model = UNet(n_channels=1, init_filters=16, n_classes=4)
    # print(model)
    # logits, out_final = model(input)
    # print(f"logits shape: {logits.shape}")
    # print(f"final output shape: {out_final.shape}")

    encoder_model  = UNetEncoder(n_channels=1, init_filters=16)
    enc_out, feats = encoder_model(input)
    print(f"enc out shape: {enc_out.shape}")
    proj_head = ProjectorHead(encoder_init_filters=16, out_dim=128)
    proj_out = proj_head(enc_out)
    print(f"projector out shape: {proj_out.shape}")    