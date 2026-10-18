
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from einops import rearrange, repeat

# Estimate Gaussian coupling width from fiber geometry and working distance.
def coupling_in_module(coreSize, inScale, z=0, wavelength=530, MFD_TO_coreDIAMETER_RATIO=1.2):

    z = z /1000
    inScale = inScale
    d_core = coreSize

    init_MFD = MFD_TO_coreDIAMETER_RATIO * d_core
    wavelength = wavelength *1e-6

    z_R = np.pi/wavelength * (init_MFD/2)**2
    w_z = init_MFD * np.sqrt(1 + (z/z_R)**2)
    sigma_z = w_z / 4  * inScale

    WIDTH_TO_SIGMA_RATIO = 6
    tmp = np.ceil(sigma_z * WIDTH_TO_SIGMA_RATIO)
    hsize = tmp if tmp % 2 == 1 else tmp+1

    return hsize.astype(int), sigma_z

def GaussKernel_at_z(coreSize, inScale, ksize, z=0, wavelength=530):
    """Returns a standard 2D Gaussian kernel array"""

    hsize, sigma = coupling_in_module(coreSize, inScale, z=z, wavelength=wavelength)

    if 'ksize' in locals():
        hsize = ksize

    mu = [0, 0]
    variance = sigma**2
    covariance = [[variance, 0], [0, variance]]

    ax = np.linspace(-(hsize-1)/2., (hsize-1)/2., hsize)
    X, Y = np.meshgrid(ax, ax)
    pos = np.dstack((X, Y))

    Z = np.exp(-0.5 * np.einsum('...k,kl,...l->...', pos - mu, np.linalg.inv(covariance), pos - mu))
    Z /= 2 * np.pi * np.sqrt(np.linalg.det(covariance))
    output = Z / np.sum(Z)

    return output

def get_patch_positions_around_cores(corePos, patch_size):
    r = np.int64(np.floor(patch_size/2))
    hw_low  = np.empty_like(corePos)
    hw_high = np.empty_like(corePos)

    hw_low  = corePos - r
    hw_high = hw_low + patch_size

    return hw_low, hw_high

# Model input coupling, optional intercore mixing, and output image formation.
class PropagaModel_MC(nn.Module):
    VALID_MODES = (
        "base_g",
        "base_r",
        "base_xt_g",
        "base_xt_r",
        "base_sc_g",
        "base_sc_r",
        "base_xt_sc_g",
        "base_xt_sc_r",
    )

    def __init__(
        self,
        corePos,
        core_global,
        coreMasks_local,
        mode="base_xt_sc_g",
        z=0,
        inScale=3000,
        pad_size=20,
        device="cuda",
    ):
        super().__init__()

        if not isinstance(mode, str) or mode not in self.VALID_MODES:
            raise ValueError(
                f"Invalid mode: {mode!r}. "
                f"Expected one of: {', '.join(self.VALID_MODES)}"
            )

        self.mode = mode
        # Select crosstalk, scattering, and kernel initialization from the mode name.
        mode_parts = mode.split("_")
        self.use_xt = "xt" in mode_parts
        self.use_sc = "sc" in mode_parts
        self.init_mode = mode_parts[-1]

        self.device = device
        self.dtype = torch.float32

        num_cores = corePos.shape[0]
        corePos = corePos.astype(np.int16)
        core_global = torch.from_numpy(core_global.astype(np.int64)).to(device)

        self.z = z
        self.corePos = corePos
        self.coreMasks_local = torch.from_numpy(coreMasks_local).to(device)
        self.num_cores = num_cores
        self.ksize_in  = coreMasks_local.shape[1]
        self.ksize_out = coreMasks_local.shape[1]

        if self.init_mode == "g":
            init_kern1 = GaussKernel_at_z(ksize=self.ksize_in,  coreSize=2/1000, inScale=inScale, z=z)
            init_kern2 = GaussKernel_at_z(ksize=self.ksize_out, coreSize=2/1000, inScale=inScale, z=0)
        else:
            init_kern1 = np.clip(np.random.normal(0.5, 0.1, (self.ksize_in, self.ksize_in)), 0, 1)
            init_kern2 = np.clip(np.random.normal(0.5, 0.1, (self.ksize_out,self.ksize_out)), 0, 1)

        ksize1 = init_kern1.shape[0]
        ksize2 = init_kern2.shape[0]
        assert ksize1 % 2 == 1
        assert ksize2 % 2 == 1

        self.kern1_size = ksize1
        self.kern2_size = ksize2

        # Precompute local patches and their correspondence to global image coordinates.
        hw1_low, hw1_high = get_patch_positions_around_cores(corePos, patch_size=self.kern1_size)
        hw1_low, hw1_high = hw1_low + pad_size, hw1_high + pad_size

        top_left_y = torch.tensor(hw1_low[:,0], dtype=torch.int16)
        top_left_x = torch.tensor(hw1_low[:,1], dtype=torch.int16)
        roi_offsets_y = torch.arange(ksize1).view(1,ksize1,1).expand(num_cores,ksize1,ksize1)
        roi_offsets_x = torch.arange(ksize1).view(1,1,ksize1).expand(num_cores,ksize1,ksize1)
        roi_y_coords = top_left_y.view(num_cores,1,1) + roi_offsets_y
        roi_x_coords = top_left_x.view(num_cores,1,1) + roi_offsets_x
        self.flat_roi_y_coords = roi_y_coords.reshape(-1)
        self.flat_roi_x_coords = roi_x_coords.reshape(-1)

        self.core_indices = core_global[:, 0]
        self.global_y_indices = core_global[:, 1]
        self.global_x_indices = core_global[:, 2]

        self.local_y_indices = torch.nonzero(self.coreMasks_local, as_tuple=False)[:, 1]
        self.local_x_indices = torch.nonzero(self.coreMasks_local, as_tuple=False)[:, 2]

        init_kern1 = rearrange(torch.tensor(init_kern1,dtype=self.dtype), 'h w -> 1 1 h w')
        init_kern2 = rearrange(torch.tensor(init_kern2,dtype=self.dtype), 'h w -> 1 1 h w')
        # Optimize a separate input and output coupling kernel for each fiber core.
        self.conv1_w = nn.ParameterList([nn.Parameter(init_kern1) for iCore in range(self.num_cores)]).to(device)
        self.conv2_w = nn.ParameterList([nn.Parameter(init_kern2) for iCore in range(self.num_cores)]).to(device)

        r1 = ksize1 // 2
        atten_mat = torch.zeros(ksize1, ksize1)
        atten_mat[r1, r1] = 1.0
        self.atten_mat = atten_mat

        self.zeroPad_layer = nn.ZeroPad2d(pad_size)
        self.coreZeroPad_layer = nn.ZeroPad2d(self.ksize_out//2)

        self.mlp = None
        if self.use_xt:
            self.mlp = nn.Sequential(
                nn.Linear(self.num_cores, self.num_cores),
                nn.ReLU(),
                nn.Linear(self.num_cores, self.num_cores),
                nn.ReLU(),
                nn.Linear(self.num_cores, self.num_cores),
                nn.ReLU(),
                nn.Linear(self.num_cores, self.num_cores),
                nn.ReLU(),
                nn.Linear(self.num_cores, self.num_cores),
            ).to(device)

        self.unet = None
        if self.use_sc:
            from .models.model_unet import UNET
            self.unet = UNET(1, 1, features=[16, 32, 64, 128], inplace=False).to(device)

    def forward(self, x):
        batch_size, img_size = x.shape[0], x.shape[2]
        num_cores = self.num_cores
        coreMasks_local = self.coreMasks_local
        ksize1 = self.kern1_size
        ksize2 = self.kern2_size
        r1 = ksize1 //2
        r2 = ksize2 //2
        atten_mat = repeat(self.atten_mat.view(1,1,ksize1,ksize1), '1 1 h w -> b 1 h w', b=batch_size).to(self.device)

        x_pad = self.zeroPad_layer(x)

        # Extract the input image patch surrounding each core.
        roi_values = x_pad[:, 0, self.flat_roi_y_coords, self.flat_roi_x_coords]
        core_in = roi_values.view(batch_size, num_cores, ksize1, ksize1)

        # Integrate input coupling to obtain one intensity value per core.
        clad_in = torch.zeros(batch_size, 0, dtype=torch.float32, device=self.device)
        for iCore in range(num_cores):
            patch_obj = core_in[:,iCore:iCore+1,:,:]
            core_in1 = F.conv2d(patch_obj, self.conv1_w[iCore], padding=r1)
            clad_in = torch.cat((clad_in, torch.sum(torch.mul(core_in1, atten_mat), dim=(2,3))), dim=1)

        # Optionally mix core intensities to represent intercore crosstalk.
        core_in2_all = clad_in
        if self.mlp is not None:
            core_in2_all = torch.add(clad_in, self.mlp(clad_in), alpha=0.1)

        # Spread each core intensity through its output coupling kernel.
        core_out1_all = torch.zeros(batch_size, 0, ksize2, ksize2, dtype=torch.float32, device=self.device)

        for iCore in range(num_cores):
            core_in3 = core_in2_all[:,iCore].view(batch_size,1,1,1)
            core_out2 = self.coreZeroPad_layer(core_in3)
            core_out1 = F.conv2d(core_out2, self.conv2_w[iCore], bias=None, padding=r2)
            core_out1_all = torch.cat((core_out1_all, core_out1),dim=1)

        flat_indices = self.global_y_indices * img_size + self.global_x_indices
        flat_batch_indices = repeat(flat_indices,'hw -> b hw',b=batch_size).to(self.device)

        rois = torch.masked_fill(core_out1_all, ~coreMasks_local, 0)
        roi_pixel_values = rois[:, self.core_indices, self.local_y_indices, self.local_x_indices]

        # Accumulate local core outputs at their global image positions.
        zeros_x = torch.zeros(batch_size, img_size**2, dtype=torch.float32, device=self.device)
        zeros_x.scatter_add_(1, flat_batch_indices, roi_pixel_values)
        cores_out = zeros_x.view(batch_size, 1, img_size, img_size)

        # Optionally add the learned cladding-scattering contribution.
        output = cores_out
        if self.unet is not None:
            output = output + self.unet(x)

        return output

