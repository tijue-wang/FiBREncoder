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

class imageDatastore_9_inputs (data.Dataset):
    def __init__(self,
                 dir_Z,
                 dir_X1,
                 dir_X2,
                 dir_X3,
                 dir_X4,
                 dir_X5,
                 dir_X6,
                 dir_X7,
                 dir_X8,
                 dir_X9,
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
        self.dir_X1 = dir_X1
        self.dir_X2 = dir_X2
        self.dir_X3 = dir_X3
        self.dir_X4 = dir_X4
        self.dir_X5 = dir_X5
        self.dir_X6 = dir_X6
        self.dir_X7 = dir_X7
        self.dir_X8 = dir_X8
        self.dir_X9 = dir_X9

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

        self.list_Z,self.list_X1,self.list_X2,self.list_X3,self.list_X4,self.list_X5,self.list_X6,self.list_X7,self.list_X8,self.list_X9 = self._scan()

        self._validate_pairs()

    def _validate_pairs(self):
        lists = [value for name, value in vars(self).items() if name.startswith("list_")]
        lengths = {len(value) for value in lists}
        if len(lengths) != 1 or not lists or not lists[0]:
            raise ValueError("Paired image directories must contain the same nonzero number of images.")
        self.num_train = len(self.list_Z)

    def __getitem__(self, idx):
        X1,X2,X3,X4,X5,X6,X7,X8,X9,Z = self._load_file(idx)

        X1,X2,X3,X4,X5,X6,X7,X8,X9,Z = common.set_channel(X1,X2,X3,X4,X5,X6,X7,X8,X9,Z, n_channels=self.n_colors)
        X1,X2,X3,X4,X5,X6,X7,X8,X9,Z = self._get_patch(X1,X2,X3,X4,X5,X6,X7,X8,X9,Z)
        X1,X2,X3,X4,X5,X6,X7,X8,X9,Z = self._augment_data(X1,X2,X3,X4,X5,X6,X7,X8,X9,Z)

        X_t1,X_t2,X_t3,X_t4,X_t5,X_t6,X_t7,X_t8,X_t9,Z_t = common.np2Tensor(X1,X2,X3,X4,X5,X6,X7,X8,X9,Z, rgb_range=self.rgb_range)
        return X_t1, X_t2, X_t3, X_t4, X_t5, X_t6, X_t7, X_t8, X_t9, Z_t

    def __len__(self):
        return self.num_train * self.repeat

    def _get_index(self, idx):
        return idx % self.num_train

    def _scan(self):
        list_Z = sorted(make_dataset(self.dir_Z))
        list_X1 = sorted(make_dataset(self.dir_X1))
        list_X2 = sorted(make_dataset(self.dir_X2))
        list_X3 = sorted(make_dataset(self.dir_X3))
        list_X4 = sorted(make_dataset(self.dir_X4))
        list_X5 = sorted(make_dataset(self.dir_X5))
        list_X6 = sorted(make_dataset(self.dir_X6))
        list_X7 = sorted(make_dataset(self.dir_X7))
        list_X8 = sorted(make_dataset(self.dir_X8))
        list_X9 = sorted(make_dataset(self.dir_X9))
        return list_Z, list_X1, list_X2, list_X3, list_X4, list_X5, list_X6, list_X7, list_X8, list_X9

    def _get_patch(self, imgX1,imgX2,imgX3,imgX4,imgX5,imgX6,imgX7,imgX8,imgX9,imgZ):
        scale = self.scale
        if self.patches:
            imgX1,imgX2,imgX3,imgX4,imgX5,imgX6,imgX7,imgX8,imgX9,imgZ = common.get_patch(
                imgX1,imgX2,imgX3,imgX4,imgX5,imgX6,imgX7,imgX8,imgX9,imgZ, patch_size=self.patch_size, scale=scale)
        elif self.patches == False and scale == 1:
            imgZ = imgZ
        else:
            ih, iw = imgX1.shape[:2]
            imgZ = imgZ[0:ih * scale, 0:iw * scale, :]
        return imgX1,imgX2,imgX3,imgX4,imgX5,imgX6,imgX7,imgX8,imgX9,imgZ

    def _augment_data(self, imgX1,imgX2,imgX3,imgX4,imgX5,imgX6,imgX7,imgX8,imgX9,imgZ):
        if self.augment:
            imgX1,imgX2,imgX3,imgX4,imgX5,imgX6,imgX7,imgX8,imgX9,imgZ = common.augment(
                imgX1,imgX2,imgX3,imgX4,imgX5,imgX6,imgX7,imgX8,imgX9,imgZ, hflip=self.hflip, vflip=self.vflip, rot=self.rotate)
        return imgX1,imgX2,imgX3,imgX4,imgX5,imgX6,imgX7,imgX8,imgX9,imgZ

    def _load_file(self, idx):
        idx = self._get_index(idx)
        if self.ext == '.npy':
            X1 = npy_loader(self.list_X1[idx])
            X2 = npy_loader(self.list_X2[idx])
            X3 = npy_loader(self.list_X3[idx])
            X4 = npy_loader(self.list_X4[idx])
            X5 = npy_loader(self.list_X5[idx])
            X6 = npy_loader(self.list_X6[idx])
            X7 = npy_loader(self.list_X7[idx])
            X8 = npy_loader(self.list_X8[idx])
            X9 = npy_loader(self.list_X9[idx])
            Z = npy_loader(self.list_Z[idx])
        else:
            X1 = default_loader(self.list_X1[idx], self.n_colors)
            X2 = default_loader(self.list_X2[idx], self.n_colors)
            X3 = default_loader(self.list_X3[idx], self.n_colors)
            X4 = default_loader(self.list_X4[idx], self.n_colors)
            X5 = default_loader(self.list_X5[idx], self.n_colors)
            X6 = default_loader(self.list_X6[idx], self.n_colors)
            X7 = default_loader(self.list_X7[idx], self.n_colors)
            X8 = default_loader(self.list_X8[idx], self.n_colors)
            X9 = default_loader(self.list_X9[idx], self.n_colors)
            Z = default_loader(self.list_Z[idx], self.n_colors)
        return X1, X2, X3, X4, X5, X6, X7, X8, X9, Z

