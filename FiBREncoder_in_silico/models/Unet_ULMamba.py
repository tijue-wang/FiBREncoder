import math

from timm.models.layers import trunc_normal_
from mamba_ssm import Mamba

import torch.nn as nn
import torch
import torch.nn.functional as F

from ..models.model_unet import UNET, UNET_multi_inputs

# Apply a shared Mamba block to four groups of spatial feature sequences.
class PVMLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state = 16, d_conv = 4, expand = 2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = Mamba(
                d_model=input_dim//4,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale= nn.Parameter(torch.ones(1))

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.input_dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)

        x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=2)
        x_mamba1 = self.mamba(x1) + self.skip_scale * x1
        x_mamba2 = self.mamba(x2) + self.skip_scale * x2
        x_mamba3 = self.mamba(x3) + self.skip_scale * x3
        x_mamba4 = self.mamba(x4) + self.skip_scale * x4
        x_mamba = torch.cat([x_mamba1, x_mamba2,x_mamba3,x_mamba4], dim=2)

        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        return out

# Use pooled features to weight channels across encoder scales.
class Channel_Att_Bridge(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()
        c_list_sum = sum(c_list) - c_list[-1]
        self.split_att = split_att
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.get_all_att = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.att1 = nn.Linear(c_list_sum, c_list[0]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[0], 1)
        self.att2 = nn.Linear(c_list_sum, c_list[1]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[1], 1)
        self.att3 = nn.Linear(c_list_sum, c_list[2]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[2], 1)
        self.att4 = nn.Linear(c_list_sum, c_list[3]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[3], 1)
        self.att5 = nn.Linear(c_list_sum, c_list[4]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[4], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, t1, t2, t3, t4, t5):
        att = torch.cat((self.avgpool(t1),
                         self.avgpool(t2),
                         self.avgpool(t3),
                         self.avgpool(t4),
                         self.avgpool(t5)), dim=1)
        att = self.get_all_att(att.squeeze(-1).transpose(-1, -2))
        if self.split_att != 'fc':
            att = att.transpose(-1, -2)
        att1 = self.sigmoid(self.att1(att))
        att2 = self.sigmoid(self.att2(att))
        att3 = self.sigmoid(self.att3(att))
        att4 = self.sigmoid(self.att4(att))
        att5 = self.sigmoid(self.att5(att))
        if self.split_att == 'fc':
            att1 = att1.transpose(-1, -2).unsqueeze(-1).expand_as(t1)
            att2 = att2.transpose(-1, -2).unsqueeze(-1).expand_as(t2)
            att3 = att3.transpose(-1, -2).unsqueeze(-1).expand_as(t3)
            att4 = att4.transpose(-1, -2).unsqueeze(-1).expand_as(t4)
            att5 = att5.transpose(-1, -2).unsqueeze(-1).expand_as(t5)
        else:
            att1 = att1.unsqueeze(-1).expand_as(t1)
            att2 = att2.unsqueeze(-1).expand_as(t2)
            att3 = att3.unsqueeze(-1).expand_as(t3)
            att4 = att4.unsqueeze(-1).expand_as(t4)
            att5 = att5.unsqueeze(-1).expand_as(t5)

        return att1, att2, att3, att4, att5

# Compute spatial attention from channel summaries.
class Spatial_Att_Bridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_conv2d = nn.Sequential(nn.Conv2d(2, 1, 7, stride=1, padding=9, dilation=3),
                                          nn.Sigmoid())

    def forward(self, t1, t2, t3, t4, t5):
        t_list = [t1, t2, t3, t4, t5]
        att_list = []
        for t in t_list:
            avg_out = torch.mean(t, dim=1, keepdim=True)
            max_out, _ = torch.max(t, dim=1, keepdim=True)
            att = torch.cat([avg_out, max_out], dim=1)
            att = self.shared_conv2d(att)
            att_list.append(att)
        return att_list[0], att_list[1], att_list[2], att_list[3], att_list[4]

# Combine spatial and channel attention for skip features.
class SC_Att_Bridge(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()

        self.catt = Channel_Att_Bridge(c_list, split_att=split_att)
        self.satt = Spatial_Att_Bridge()

    def forward(self, t1, t2, t3, t4, t5):
        r1, r2, r3, r4, r5 = t1, t2, t3, t4, t5

        satt1, satt2, satt3, satt4, satt5 = self.satt(t1, t2, t3, t4, t5)
        t1, t2, t3, t4, t5 = satt1 * t1, satt2 * t2, satt3 * t3, satt4 * t4, satt5 * t5

        r1_, r2_, r3_, r4_, r5_ = t1, t2, t3, t4, t5
        t1, t2, t3, t4, t5 = t1 + r1, t2 + r2, t3 + r3, t4 + r4, t5 + r5

        catt1, catt2, catt3, catt4, catt5 = self.catt(t1, t2, t3, t4, t5)
        t1, t2, t3, t4, t5 = catt1 * t1, catt2 * t2, catt3 * t3, catt4 * t4, catt5 * t5

        return t1 + r1_, t2 + r2_, t3 + r3_, t4 + r4_, t5 + r5_

# Encode and decode features using lightweight Mamba blocks.
class UltraLight_VM_UNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=8, c_list=[8,16,24,32,48,64], split_att='fc', bridge=True, sigm=True):
        super().__init__()

        self.bridge = bridge

        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1),
        )
        self.encoder2 =nn.Sequential(
            nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1),
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1),
        )
        self.encoder4 = nn.Sequential(
            PVMLayer(input_dim=c_list[2], output_dim=c_list[3])
        )
        self.encoder5 = nn.Sequential(
            PVMLayer(input_dim=c_list[3], output_dim=c_list[4])
        )
        self.encoder6 = nn.Sequential(
            PVMLayer(input_dim=c_list[4], output_dim=c_list[5])
        )

        if bridge:
            self.scab = SC_Att_Bridge(c_list, split_att)
            print('SC_Att_Bridge was used')

        self.decoder1 = nn.Sequential(
            PVMLayer(input_dim=c_list[5], output_dim=c_list[4])
        )
        self.decoder2 = nn.Sequential(
            PVMLayer(input_dim=c_list[4], output_dim=c_list[3])
        )
        self.decoder3 = nn.Sequential(
            PVMLayer(input_dim=c_list[3], output_dim=c_list[2])
        )
        self.decoder4 = nn.Sequential(
            nn.Conv2d(c_list[2], c_list[1], 3, stride=1, padding=1),
        )
        self.decoder5 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1),
        )
        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

        self.apply(self._init_weights)

        self.sigm = sigm

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        out = F.gelu(F.max_pool2d(self.ebn1(x),2,2))
        t1 = out

        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
        t2 = out

        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
        t3 = out

        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)),2,2))
        t4 = out

        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)),2,2))
        t5 = out

        if self.bridge: t1, t2, t3, t4, t5 = self.scab(t1, t2, t3, t4, t5)

        out = F.gelu(self.encoder6(out))

        out5 = F.gelu(self.dbn1(self.decoder1(out)))
        out5 = torch.add(out5, t5)

        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)),scale_factor=(2,2),mode ='bilinear',align_corners=True))
        out4_padded = torch.zeros_like(t4)
        out4_padded[:, :, :out4.size(2), :out4.size(3)] = out4
        out4 = torch.add(out4_padded, t4)

        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)),scale_factor=(2,2),mode ='bilinear',align_corners=True))
        out3 = torch.add(out3, t3)

        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)),scale_factor=(2,2),mode ='bilinear',align_corners=True))
        out2 = torch.add(out2, t2)

        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)),scale_factor=(2,2),mode ='bilinear',align_corners=True))
        out1 = torch.add(out1, t1)

        out0 = F.interpolate(self.final(out1),scale_factor=(2,2),mode ='bilinear',align_corners=True)

        final_out = torch.sigmoid(out0) if self.sigm else out0
        return  final_out

# Pass U-Net features to the lightweight Mamba refinement network.
class Unet_ULMamba(nn.Module):
    def __init__(self, c_unet=[32, 64, 128, 256], c_ulm=[8,16,24,32,48,64]):
        super(Unet_ULMamba, self).__init__()

        self.unet = UNET(out_channels=c_ulm[0], features=c_unet, inplace=False)

        self.ulmamba = UltraLight_VM_UNet(num_classes=1, input_channels=1, c_list=c_ulm)

    def forward(self, x):

        unet_output = self.unet(x)

        final_output = self.ulmamba(unet_output)

        return final_output

from ..models.spatialTransformer import spatialTransformer

# Align the input with a spatial transformer before reconstruction.
class Unet_ULMamba_st(nn.Module):
    def __init__(self):
        super(Unet_ULMamba_st, self).__init__()

        self.spatialTransformer = spatialTransformer(input_size=960)

        self.unet = UNET(out_channels=8, inplace=False)
        self.ulmamba = UltraLight_VM_UNet(num_classes=1, input_channels=1)

    def forward(self, x):

        x = self.spatialTransformer(x)

        unet_output = self.unet(x)

        final_output = self.ulmamba(unet_output)

        return final_output

from ..models.spatialTransformer import affine2d
# Apply a learned affine transform before reconstruction.
class Unet_ULMamba_affine(nn.Module):
    def __init__(self, features=[16,32,64,128]):
        super(Unet_ULMamba_affine, self).__init__()

        self.affine2d = affine2d()

        self.unet = UNET(features=features,out_channels=8, inplace=False)
        self.ulmamba = UltraLight_VM_UNet(num_classes=1, input_channels=1)

    def forward(self, x):

        x = self.affine2d(x)

        unet_output = self.unet(x)

        final_output = self.ulmamba(unet_output)

        return final_output

# Fuse multiple input images with U-Net, then refine with Mamba.
class Unet_ULMamba_multi_inputs(nn.Module):
    def __init__(self, num_inputs=5, c_unet=[32, 64, 128, 256], c_ulm=[8,16,24,32,48,64]):
        super(Unet_ULMamba_multi_inputs, self).__init__()
        self.num_inputs = num_inputs

        self.unet = UNET_multi_inputs(out_channels=c_ulm[0], num_inputs=num_inputs, features=c_unet, inplace=False)

        self.ulmamba = UltraLight_VM_UNet(num_classes=1, input_channels=1, c_list=c_ulm)

    def forward(self, *inputs):
        assert len(inputs) == self.num_inputs, "Number of inputs must match num_inputs"

        unet_out = self.unet(*inputs)
        final_out = self.ulmamba(unet_out)
        return final_out

# Reconstruct features and increase resolution with pixel shuffling.
class Unet_ULMamba_x2(nn.Module):
    def __init__(self, features=[32, 64, 128, 256], factor=2):
        super(Unet_ULMamba_x2, self).__init__()

        self.unet = UNET(out_channels=8, features=features, inplace=False)

        self.ulmamba = UltraLight_VM_UNet(num_classes=factor**2, input_channels=1)

        self.upsampler = nn.PixelShuffle(factor)

    def forward(self, x):

        unet_output = self.unet(x)

        ulmamba_output = self.ulmamba(unet_output)

        final_output = self.upsampler(ulmamba_output)
        return final_output

def icnr_init(tensor, upscale_factor=2, initializer=nn.init.kaiming_normal_):
    """ICNR Initialization: Make the initial output of the convolutional layer
    before PixelShuffle close to the nearest neighbor interpolation."""
    new_shape = [int(tensor.shape[0] / (upscale_factor ** 2))] + list(tensor.shape[1:])
    subkernel = torch.zeros(new_shape)
    initializer(subkernel)
    subkernel = subkernel.repeat_interleave(upscale_factor ** 2, dim=0)
    tensor.data.copy_(subkernel)

# Combine multi-input reconstruction with subpixel upsampling.
class Unet_ULMamba_multi_inputs_UpSp(nn.Module):
    def __init__(self, num_inputs=5, out_channels=8, c_unet=[32,64,128,256], c_ulm=[8,16,24,32,48,64], UpFactor=2, sigm=True):
        super(Unet_ULMamba_multi_inputs_UpSp, self).__init__()

        self.num_inputs = num_inputs

        self.unet = UNET_multi_inputs(out_channels=c_ulm[0], num_inputs=num_inputs, features=c_unet, inplace=False)

        self.ulmamba = UltraLight_VM_UNet(num_classes=out_channels, input_channels=c_ulm[1], c_list=c_ulm, sigm=sigm)

        self.upsampler = nn.Sequential(*[
            nn.Conv2d(out_channels, UpFactor**2, 3, padding=1, bias=False),
            nn.PixelShuffle(UpFactor),
            nn.Conv2d(1, 1, 3, padding=1),
            ])

        icnr_init(self.upsampler[0].weight, upscale_factor=UpFactor)

    def forward(self, *inputs):
        assert len(inputs) == self.num_inputs, "Number of inputs must match num_inputs"

        unet_out = self.unet(*inputs)

        ulmamba_output = self.ulmamba(unet_out)
        final_out = self.upsampler(ulmamba_output)

        return final_out

