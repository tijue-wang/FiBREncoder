import torch
from torch import nn
import torch.nn.functional as F

from einops import repeat

# Predict an affine alignment from the input image.
class spatialTransformer(nn.Module):
    def __init__(self, input_size):
        super(spatialTransformer, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()

        self.input_size = input_size
        assert input_size % 4 == 0
        self.feature_size = int(input_size/4 - 4)

        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(10 * self.feature_size**2, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * self.feature_size**2)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, mode='bicubic', align_corners=True)

        return x

# Apply a directly optimized affine transform to each batch image.
class affine2d(nn.Module):
    def __init__(self):
        super(affine2d, self).__init__()

        self.matrix = nn.Parameter(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        matrix = repeat(self.matrix.view(2, 3), 'h w -> b h w', b=x.shape[0])

        grid = F.affine_grid(matrix, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, mode='bicubic', align_corners=True)
        return x

# Learn an image homography with separately controllable parameter groups.
class perspective2d(nn.Module):
    """
    Learnable 2D homography transformation layer.

    The homography matrix is parameterized as H = I + Delta:
        H = [[1+d0,  d1,  d2],        <- affine_delta[0..2]
             [ d3, 1+d4,  d5],        <- affine_delta[3..5]
             [ d6,   d7,   1]]        <- persp_delta[0..1]

    Input / Output:  (B, C, H, W) -> (B, C, H, W)

    Uses grid_sample for inverse warping: for every pixel in the output
    (registered meas), H tells us where to sample in the input (meas).
    """

    def __init__(self, height: int, width: int):
        super().__init__()
        self.affine_delta = nn.Parameter(torch.zeros(6))
        self.persp_delta  = nn.Parameter(torch.zeros(2))

        ys = torch.linspace(-1, 1, height)
        xs = torch.linspace(-1, 1, width)
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        base = torch.stack(
            [gx.flatten(), gy.flatten(), torch.ones(height * width)], dim=1
        )
        self.register_buffer('base_grid', base)
        self.height, self.width = height, width

    def get_matrix(self) -> torch.Tensor:
        d, p = self.affine_delta, self.persp_delta
        row0 = torch.stack([1 + d[0], d[1],     d[2]])
        row1 = torch.stack([d[3],     1 + d[4], d[5]])
        row2 = torch.cat([p, p.new_ones(1)])
        return torch.stack([row0, row1, row2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B  = x.shape[0]
        H  = self.get_matrix()
        pts = self.base_grid @ H.t()

        xy = pts[:, :2] / pts[:, 2:].clamp(min=1e-8)

        grid = xy.view(1, self.height, self.width, 2).expand(B, -1, -1, -1)
        return F.grid_sample(
            x, grid, mode='bilinear',
            padding_mode='zeros', align_corners=True
        )

    def freeze_persp(self):
        """Freeze perspective params; train affine only."""
        self.persp_delta.requires_grad_(False)
        self.affine_delta.requires_grad_(True)

    def freeze_affine(self):
        """Freeze affine params; train perspective only."""
        self.affine_delta.requires_grad_(False)
        self.persp_delta.requires_grad_(True)

    def unfreeze_all(self):
        """Unfreeze all homography parameters."""
        self.affine_delta.requires_grad_(True)
        self.persp_delta.requires_grad_(True)

