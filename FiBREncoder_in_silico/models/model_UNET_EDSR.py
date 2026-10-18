import math
import torch
import torch.nn as nn

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

# Learn a residual correction while retaining the input features.
class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

# Increase spatial resolution using pixel shuffling.
class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

def Conv2d_stride2(kernel_size, dilation):
    padding = math.ceil(0.5*(kernel_size*dilation-dilation-2+1))
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

        x1 = self.convBlock1(x)
        skip = self.skip(x)
        x1 += skip

        return torch.add(x1, self.convBlock2(x1))

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

    def forward(self, x):

        x1 = self.convBlock1(x)
        skip = self.skip(x)
        x1 += skip

        return torch.add(x1, self.convBlock2(x1))

# Encode spatial features and recover image detail through the decoder.
class UNET(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256], outScale=1, inplace=True):
        super(UNET, self).__init__()
        self.unetEncoder = nn.ModuleList()
        self.unetDecoder = nn.ModuleList()

        Feats = features

        self.ConvIn = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, padding='same', bias=True),
            nn.ReLU(inplace=inplace),
            )

        self.unetEncoder.append(DownBlock(8, Feats[0], kernel_size=9, dilation=3, inplace=inplace))
        self.unetEncoder.append(DownBlock(Feats[0], Feats[1], kernel_size=3, dilation=1, inplace=inplace))
        self.unetEncoder.append(DownBlock(Feats[1], Feats[2], kernel_size=3, dilation=1, inplace=inplace))
        self.unetEncoder.append(DownBlock(Feats[2], Feats[3], kernel_size=3, dilation=1, inplace=inplace))

        revFeats = features[::-1]
        self.unetDecoder.append(UpBlock(revFeats[0], revFeats[1], kernel_size=3, inplace=inplace))
        self.unetDecoder.append(UpBlock(revFeats[1], revFeats[2], kernel_size=3, inplace=inplace))
        if outScale <= 2:
            self.unetDecoder.append(UpBlock(revFeats[2], revFeats[3], kernel_size=3, inplace=inplace))
            ConvOut_in = revFeats[3]
            if outScale == 1:
                self.unetDecoder.append(UpBlock(revFeats[3], out_channels=out_channels, kernel_size=3, inplace=inplace))
                ConvOut_in = out_channels

        self.ConvOut = nn.Conv2d(ConvOut_in, out_channels, kernel_size=3, padding='same', bias=True)

    def forward(self, x):
        skip_connects = []

        x = self.ConvIn(x)

        for downblock in self.unetEncoder:
            x = downblock(x)
            skip_connects.append(x)

        skip_connects.pop(-1)
        skip_connects = skip_connects[::-1]

        for ii in range(0, len(self.unetDecoder)-1):
            x = self.unetDecoder[ii](x)
            x = torch.add(x, skip_connects[ii])

        x = self.unetDecoder[-1](x)
        return self.ConvOut(x)

# Refine image features with a stack of residual blocks.
class EDSR(nn.Module):
    def __init__(self, in_channels=1, n_colors=1, n_resblocks=32, n_feats=256, scale=2, conv=default_conv, rgb_range=1):
        super(EDSR, self).__init__()

        in_channels = in_channels
        n_resblocks = n_resblocks
        n_feats = n_feats
        scale = scale
        n_colors = n_colors
        rgb_range = rgb_range

        kernel_size = 3
        act = nn.ReLU(True)
        res_scale = 1

        m_head = [conv(in_channels, n_feats, kernel_size)]

        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        m_tail = []
        if scale >= 2:
            m_tail.append(Upsampler(conv, scale, n_feats, act=False))
        m_tail.append(conv(n_feats, n_colors, kernel_size))

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)

        return x

# Extract features with U-Net, then refine them with EDSR.
class UNET_EDSR(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256], n_resblocks=32, n_feats=256, scale=1,
                 n_colors=1, rgb_range=1, inplace=True):
        super(UNET_EDSR, self).__init__()

        in_channels = in_channels
        out_channels = out_channels
        n_colors = n_colors
        rgb_range = rgb_range
        features = features
        inplace = inplace
        n_resblocks = n_resblocks
        n_feats = n_feats
        scale = scale

        from ..models.model_unet import UNET as unet_256
        self.unet = unet_256(in_channels=in_channels, out_channels=8, features=features, inplace=False)

        self.edsr = EDSR(in_channels=8, n_colors=1, n_resblocks=n_resblocks, n_feats=n_feats, scale=scale, rgb_range=1)

    def forward(self, x):

        x = self.unet(x)

        x = self.edsr(x)

        return x

