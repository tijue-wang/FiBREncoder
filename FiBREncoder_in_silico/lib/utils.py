import os, time
import numpy as np
import random
from collections import OrderedDict
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import torch
from torch import nn

def setup_seed(seed=42, ConvOptim=False):
    if seed is None:
        seed = random.randint(1, 10000)
        print("Ramdom Seed: ", seed)

    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if ConvOptim:
         torch.backends.cudnn.benchmark = False
         torch.backends.cudnn.deterministic = True

def save_checkpoint(model, model_arch, optimizer, epoch, loss_stats, args, text=''):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": unwrap_module(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "loss_stats": loss_stats,
        "epoch": epoch,
        "args": args,
        }
    if args.save_model_arch:
        checkpoint.update({"model_arch": model_arch})
    torch.save(checkpoint, args.save_path + "net_epoch_{}".format(epoch) + text + ".pth.tar")

def load_checkpoint(checkpoint, model):
    if isinstance(model, nn.DataParallel):
        print("=> Loading checkpoint with data parallel")
        state_dict = fix_state_dict_for_loading(checkpoint["state_dict"], to_dataparallel=1)
        model.load_state_dict(state_dict, strict=True)
    else:
        print("=> Loading checkpoint")
        model.load_state_dict(checkpoint["state_dict"])

def adjust_learning_rate(optimizer, epoch, lr_init, gamma, step_size):
    factor = epoch // step_size
    lr = lr_init * (gamma ** factor)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def tensor2np(tensor, out_type=np.uint8, min_max=(0, 1)):
    tensor = tensor.float().detach().cpu().clamp_(*min_max)
    img_np = tensor.numpy()
    img_np = np.transpose(img_np, (1, 2, 0))
    if out_type == np.uint8 and min_max==(0, 1):
        img_np = (img_np * 255.0).round()
    else:
        img_np = img_np.round()

    return img_np.astype(out_type)

class Timer():
    def __init__(self):
        self.v = time.time()
    def s(self):
        self.v = time.time()
    def t(self):
        return time.time() - self.v

def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)

def _get_matlab_like_data_range(im1, im2):
    """
    Approximate MATLAB's DynamicRange / PeakVal behavior for psnr/ssim:
    - uint8  -> 255
    - uint16 -> 65535
    - float images in [0, 1] -> 1.0
    - float images that are actually in [0, 255] -> 255.0
    """
    im1 = np.asarray(im1)
    im2 = np.asarray(im2)

    if np.issubdtype(im1.dtype, np.integer):
        info = np.iinfo(im1.dtype)
        return float(info.max) - float(info.min)

    if np.issubdtype(im2.dtype, np.integer):
        info = np.iinfo(im2.dtype)
        return float(info.max) - float(info.min)

    max_val = max(float(np.max(im1)), float(np.max(im2)))
    min_val = min(float(np.min(im1)), float(np.min(im2)))

    if min_val >= 0.0 and max_val <= 1.0:
        return 1.0

    if min_val >= 0.0 and max_val <= 255.0:
        return 255.0

    return max_val - min_val

def compute_psnr(ref, im):
    data_range = _get_matlab_like_data_range(ref, ref)
    return psnr(ref, im, data_range=data_range)

def compute_ssim(im1, im2):
    im1 = np.asarray(im1); im2 = np.asarray(im2)
    assert im1.shape == im2.shape
    channel_axis = None if im1.ndim == 2 else -1
    data_range = _get_matlab_like_data_range(im1, im2)

    return ssim(
        im1, im2,
        channel_axis=channel_axis,
        data_range=data_range,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
        K1=0.01,
        K2=0.03,
    )

def unwrap_module(m):
    return m.module if isinstance(m, nn.DataParallel) else m

def fix_state_dict_for_loading(sd, to_dataparallel: bool):
    """Normalize checkpoint keys for DataParallel loading."""
    new_sd = OrderedDict()
    for k, v in sd.items():
        if to_dataparallel and not k.startswith("module."):
            new_sd["module."+k] = v
        elif (not to_dataparallel) and k.startswith("module."):
            new_sd[k[len("module."):]] = v
        else:
            new_sd[k] = v
    return new_sd

from ..models.spatialTransformer import perspective2d
class HomographyStageScheduler:
    """
    Manages three-stage training of the homography layer.

    Stage 1  [affine_only]:   Train affine 6-DoF only.
    Stage 2  [persp_only]:    Train perspective 2-DoF only.
    Stage 3  [joint]:         Fine-tune both simultaneously.

    Args:
        homography_layer: The LearnableHomography module.
        total_epochs:     Total number of training epochs.
        stage_ratios:     Length-3 list/tuple that sums to 1.0,
                          e.g. [0.4, 0.4, 0.2] means 40% affine,
                          40% perspective, 20% joint fine-tuning.

    Usage:
        scheduler = HomographyStageScheduler(layer, 100, [0.4, 0.4, 0.2])
        for epoch in range(1, 101):
            scheduler.step(epoch)   # call once at the top of each epoch
    """

    STAGE_AFFINE = 1
    STAGE_PERSP  = 2
    STAGE_JOINT  = 3

    _STAGE_NAMES = {
        STAGE_AFFINE: "Stage 1 - Affine only (6 DoF)",
        STAGE_PERSP:  "Stage 2 - Perspective only (2 DoF)",
        STAGE_JOINT:  "Stage 3 - Joint fine-tune (8 DoF)",
    }

    _STAGE_ACTIONS = {
        STAGE_AFFINE: perspective2d.freeze_persp,
        STAGE_PERSP:  perspective2d.freeze_affine,
        STAGE_JOINT:  perspective2d.unfreeze_all,
    }

    def __init__(
        self,
        homography_layer: perspective2d,
        total_epochs: int,
        stage_ratios: tuple | list = (0.4, 0.4, 0.2),
    ):
        assert len(stage_ratios) == 3, "stage_ratios must have exactly 3 elements"
        assert abs(sum(stage_ratios) - 1.0) < 1e-6, "stage_ratios must sum to 1"

        self.layer = homography_layer
        self.current_stage = None

        cum = 0
        self._boundaries = []
        for r in stage_ratios:
            cum += r
            self._boundaries.append(int(round(cum * total_epochs)))

    def _epoch_to_stage(self, epoch: int) -> int:
        """Map a 1-indexed epoch to its stage id."""
        if epoch <= self._boundaries[0]:
            return self.STAGE_AFFINE
        elif epoch <= self._boundaries[1]:
            return self.STAGE_PERSP
        else:
            return self.STAGE_JOINT

    def step(self, epoch: int) -> bool:
        """
        Call at the start of each epoch (1-indexed).
        Returns True when a stage transition occurs.
        """
        target = self._epoch_to_stage(epoch)

        if target == self.current_stage:
            return False

        self.current_stage = target
        self._STAGE_ACTIONS[target](self.layer)
        return True

    @property
    def stage_name(self) -> str:
        """Human-readable name of the current stage."""
        return self._STAGE_NAMES.get(self.current_stage, "Unknown")

