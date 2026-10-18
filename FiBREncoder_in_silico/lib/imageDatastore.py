import torch.utils.data as data
import os
import cv2
import numpy as np
from ..lib import common

def default_loader(path, n_colors):
    if n_colors == 3:
        return cv2.imread(path, cv2.IMREAD_UNCHANGED)[:, :, [2, 1, 0]]
    else:
        return cv2.imread(path, cv2.IMREAD_UNCHANGED)

def npy_loader(path):
    return np.load(path)

IMG_EXTENSIONS = ['.png', '.npy', ]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images

class imageDatastore(data.Dataset):
    def __init__(self,
                 dir_Z,
                 dir_X,
                 patches = False,
                 patch_size = 256,
                 n_colors = 1,
                 rgb_range = 1,
                 augment = False,
                 hflip = False,
                 vflip = False,
                 rotate = False,
                 ext = ".png"):

        self.dir_Z = dir_Z
        self.dir_X = dir_X
        self.patches = patches

        self.patch_size = patch_size
        self.n_colors = n_colors
        self.rgb_range = rgb_range
        self.ext = ext

        self.augment = True if hflip == True or vflip == True or rotate == True else False
        self.hflip = hflip
        self.vflip = vflip
        self.rotate = rotate

        self.num_train = len(os.listdir(self.dir_Z))
        self.scale = 1

        self.repeat = 1

        self.list_Z, self.list_X = self._scan()

        self._validate_pairs()

    def _validate_pairs(self):
        lists = [value for name, value in vars(self).items() if name.startswith("list_")]
        lengths = {len(value) for value in lists}
        if len(lengths) != 1 or not lists or not lists[0]:
            raise ValueError("Paired image directories must contain the same nonzero number of images.")
        self.num_train = len(self.list_Z)

    def __getitem__(self, idx):
        X, Z = self._load_file(idx)

        X, Z = common.set_channel(X, Z, n_channels=self.n_colors)
        X, Z = self._get_patch(X, Z)
        X, Z = self._augment_data(X,Z)

        X_tensor, Z_tensor = common.np2Tensor(X, Z, rgb_range=self.rgb_range)
        return X_tensor, Z_tensor

    def __len__(self):
        return self.num_train * self.repeat

    def _get_index(self, idx):
        return idx % self.num_train

    def _scan(self):
        list_Z = sorted(make_dataset(self.dir_Z))
        list_X = sorted(make_dataset(self.dir_X))
        return list_Z, list_X

    def _get_patch(self, imgX, imgZ):
        scale = self.scale
        if self.patches:
            imgX, imgZ = common.get_patch(imgX, imgZ, patch_size=self.patch_size, scale=scale)
        elif self.patches == False and scale == 1:
            imgZ = imgZ
        else:
            ih, iw = imgX.shape[:2]
            imgZ = imgZ[0:ih * scale, 0:iw * scale, :]
        return imgX, imgZ

    def _augment_data(self, imgX, imgZ):
        if self.augment:
            imgX, imgZ = common.augment(imgX, imgZ, hflip=self.hflip, vflip=self.vflip, rot=self.rotate)
        return imgX, imgZ

    def _load_file(self, idx):
        idx = self._get_index(idx)
        if self.ext == '.npy':
            X = npy_loader(self.list_X[idx])
            Z = npy_loader(self.list_Z[idx])
        else:
            X = default_loader(self.list_X[idx], self.n_colors)
            Z = default_loader(self.list_Z[idx], self.n_colors)
        return X, Z
