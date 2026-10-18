import os, time
import numpy as np
import random
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import h5py
from einops import rearrange

import torch

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
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loss_stats": loss_stats,
        "epoch": epoch,
        "args": args,
        }
    if args.save_model_arch:
        checkpoint.update({"model_arch": model_arch})
    torch.save(checkpoint, args.save_path + "net_epoch_{}".format(epoch) + text + ".pth.tar")

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def adjust_learning_rate(optimizer, epoch, lr_init, gamma, step_size):
    factor = epoch // step_size
    lr = lr_init * (gamma ** factor)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def tensor2np(tensor, out_type=np.uint8, min_max=(0, 1)):
    tensor = tensor.float().cpu().clamp_(*min_max)
    img_np = tensor.numpy()
    img_np = np.transpose(img_np, (1, 2, 0))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()

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

def save_kernels_to_h5(model, epoch, save_dir, prefix='kernels'):
    """
    Save model.conv1_w and model.conv2_w to an h5 file every N epochs.
    Storage format: [h, w, num_cores], dtype=float16.

    Args:
        model        : Model containing conv1_w / conv2_w (nn.ParameterList or nn.Parameter)
        epoch        : Current epoch (int)
        save_dir     : Directory to save the h5 files
        save_every_n : Save once every N epochs
        prefix       : Filename prefix
    """

    os.makedirs(save_dir, exist_ok=True)

    def _stack_params(param_container):
        """
        Convert ParameterList([1,1,h,w] * num_cores) or a single Parameter[1,num_cores,h,w]
        into a numpy array of shape [h, w, num_cores] in float16.
        """
        if isinstance(param_container, torch.nn.ParameterList) or isinstance(param_container, list):
            tensors = [p.detach().cpu() for p in param_container]
            stacked = torch.stack(tensors, dim=0)
            stacked = stacked.squeeze(1).squeeze(1)
            arr = rearrange(stacked, 'c h w -> h w c').numpy()
        else:
            t = param_container.detach().cpu()
            if t.dim() == 4 and t.shape[0] == 1:
                t = t.squeeze(0)
            arr = rearrange(t, 'c h w -> h w c').numpy()
        return arr.astype(np.float16)

    conv1_arr = _stack_params(model.conv1_w)
    conv2_arr = _stack_params(model.conv2_w)

    fname = os.path.join(save_dir, f'{prefix}_epoch_{epoch:04d}.h5')
    with h5py.File(fname, 'w') as f:
        f.create_dataset('conv1_w', data=conv1_arr,
                         compression='gzip', compression_opts=9)
        f.create_dataset('conv2_w', data=conv2_arr,
                         compression='gzip', compression_opts=9)
        f.attrs['epoch']     = epoch
        f.attrs['dtype']     = 'float16'
        f.attrs['shape_fmt'] = 'h, w, num_cores'

    print(f'[Epoch {epoch}] kernels saved -> {fname}')

def save_valid_preds_h5(preds, targets=None, epoch=None,
                        save_dir='./saved_valid_preds',
                        prefix='valid',
                        psnr=None, ssim=None, loss=None):
    """
    Save validation predictions (and optionally targets) into an h5 file
    using float16 to halve the disk footprint compared to float32.

    Args:
        preds    : list of numpy arrays OR a single stacked numpy array
                   each sample has shape [H, W] (or arbitrary trailing dims)
        targets  : same format as preds, or None to skip
        epoch    : current epoch (used in the output filename, optional)
        save_dir : output directory
        prefix   : filename prefix
        psnr/ssim/loss : optional metrics, stored as h5 attributes
    """
    if preds is None or len(preds) == 0:
        return None

    os.makedirs(save_dir, exist_ok=True)

    preds_arr = np.stack(preds, axis=0) if isinstance(preds, list) else preds
    preds_arr = preds_arr.astype(np.float16)

    epoch_tag = f'_epoch_{epoch:04d}' if epoch is not None else ''
    fname = os.path.join(save_dir, f'{prefix}{epoch_tag}.h5')

    with h5py.File(fname, 'w') as f:
        f.create_dataset('pred', data=preds_arr,
                         compression='gzip', compression_opts=9)

        if targets is not None and len(targets) > 0:
            tgts_arr = np.stack(targets, axis=0) if isinstance(targets, list) else targets
            tgts_arr = tgts_arr.astype(np.float16)
            f.create_dataset('target', data=tgts_arr,
                             compression='gzip', compression_opts=9)

        if epoch is not None:
            f.attrs['epoch'] = epoch
        f.attrs['dtype'] = 'float16'
        if psnr is not None: f.attrs['psnr'] = float(psnr)
        if ssim is not None: f.attrs['ssim'] = float(ssim)
        if loss is not None: f.attrs['loss'] = float(loss)

    print(f'[Epoch {epoch}] valid predictions saved -> {fname}')
    return fname

