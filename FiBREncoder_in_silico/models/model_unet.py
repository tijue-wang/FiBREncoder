import torch
import torch.nn as nn
from math import ceil

def Conv2d_stride2(kernel_size, dilation):
    padding = ceil(0.5*(kernel_size*dilation-dilation-2+1))
    return padding

# Reduce spatial resolution while extracting local features.
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, inplace=True):
        super(DownBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation

        padding = Conv2d_stride2(self.kernel_size, self.dilation)

        self.convBlock1 = nn.Sequential(
                nn.ReLU(inplace=inplace),
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=padding, dilation=dilation, bias=False),

                nn.ReLU(inplace=inplace),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same', bias=True),)

        self.convBlock2 = nn.Sequential(
                nn.ReLU(inplace=inplace),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same', bias=True),

                nn.ReLU(inplace=inplace),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same', bias=True),)

        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=0, bias=True)

    def forward(self, x):
        skip = self.skip(x)
        x1 = self.convBlock1(x)
        x1 += skip

        return x1 + self.convBlock2(x1)

# Upsample features and combine them with the supplied skip features.
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, inplace=True):
        super(UpBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.convBlock1 = nn.Sequential(
                nn.ReLU(inplace=inplace),
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=True),

                nn.ReLU(inplace=inplace),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same', bias=True),)

        self.convBlock2 = nn.Sequential(
                nn.ReLU(inplace=inplace),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same', bias=True),

                nn.ReLU(inplace=inplace),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same', bias=True),)

        self.skip = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=True)

    def forward(self, x, feature_map):
        x += feature_map
        skip = self.skip(x)
        x1 = self.convBlock1(x)
        x1 += skip
        return x1 + self.convBlock2(x1)

# Encode spatial features and recover image detail through the decoder.
class UNET(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256], inplace=True):
        super(UNET, self).__init__()
        Feats = features
        revFeats = features[::-1]

        self.ConvIn = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, padding='same', bias=True),
            nn.ReLU(inplace=inplace),)

        self.down1 = DownBlock(8, Feats[0], kernel_size=9, dilation=3, inplace=inplace)
        self.down2 = DownBlock(Feats[0], Feats[1], kernel_size=3, dilation=1, inplace=inplace)
        self.down3 = DownBlock(Feats[1], Feats[2], kernel_size=3, dilation=1, inplace=inplace)
        self.down4 = DownBlock(Feats[2], Feats[3], kernel_size=3, dilation=1, inplace=inplace)

        self.up1 = UpBlock(revFeats[0], revFeats[1], kernel_size=3, inplace=inplace)
        self.up2 = UpBlock(revFeats[1], revFeats[2], kernel_size=3, inplace=inplace)
        self.up3 = UpBlock(revFeats[2], revFeats[3], kernel_size=3, inplace=inplace)
        self.up4 = UpBlock(revFeats[3], 8, kernel_size=3, inplace=inplace)

        self.ConvOut = nn.Conv2d(in_channels=8, out_channels=out_channels, kernel_size=3, padding ='same',bias=True)

    def forward(self, x):
        x1 = self.ConvIn(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5,x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)

        return self.ConvOut(x)

# Extract features from one input before multi-input fusion.
class SharedModule(nn.Module):
    def __init__(self, in_channels, features, inplace=False):
        super(SharedModule, self).__init__()

        self.ConvIn = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, padding='same', bias=True),
            nn.ReLU(inplace=inplace),
            )

        self.down1 = DownBlock(8, features[0], kernel_size=9, dilation=3, inplace=inplace)

    def forward(self, x):
        x = self.down1(self.ConvIn(x))
        return x

# Fuse features from multiple acquisitions before the common decoder.
class UNET_multi_inputs(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_inputs=5, features=[32, 64, 128, 256], inplace=False):
        super(UNET_multi_inputs, self).__init__()

        self.num_inputs = num_inputs
        Feats = features
        revFeats = features[::-1]

        self.shared_modules = nn.ModuleList([
            SharedModule(in_channels=in_channels, features=Feats, inplace=inplace) for _ in range(num_inputs)
        ])

        self.down2 = DownBlock(Feats[0], Feats[1], kernel_size=3, dilation=1, inplace=inplace)
        self.down3 = DownBlock(Feats[1], Feats[2], kernel_size=3, dilation=1, inplace=inplace)
        self.down4 = DownBlock(Feats[2], Feats[3], kernel_size=3, dilation=1, inplace=inplace)

        self.up1 = UpBlock(revFeats[0], revFeats[1], kernel_size=3, inplace=inplace)
        self.up2 = UpBlock(revFeats[1], revFeats[2], kernel_size=3, inplace=inplace)
        self.up3 = UpBlock(revFeats[2], revFeats[3], kernel_size=3, inplace=inplace)
        self.up4 = UpBlock(revFeats[3], 8, kernel_size=3, inplace=inplace)

        self.ConvOut = nn.Conv2d(in_channels=8, out_channels=out_channels, kernel_size=3, padding='same',bias=True)

    def forward(self, *inputs):
        assert len(inputs) == self.num_inputs, "Number of inputs must match num_inputs"

        x2 = self.shared_modules[0](inputs[0])
        for i in range(1, self.num_inputs):
            x2 = torch.add(x2, self.shared_modules[i](inputs[i]))

        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5,x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)

        return self.ConvOut(x)

