import os
import random

import nibabel as nib
import numpy as np
from torch.utils.data import Dataset
from os import walk
from skimage import transform
import torchvision.transforms as T
import torch
from Dataloader.utils import CustomCompose, SimpleRandomRotation, ElasticDeformation

vol_transform = T.Compose([
    T.RandomRotation(15),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomResizedCrop((192, 192), scale=(0.95, 1.05), interpolation=T.InterpolationMode.NEAREST),
    SimpleRandomRotation(),
    T.ColorJitter(brightness=[0.7, 1.3], contrast=[0.7, 1.3]),
])

seg_transform = T.Compose([
    T.RandomRotation(15),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomResizedCrop((192, 192), scale=(0.95, 1.05), interpolation=T.InterpolationMode.NEAREST),
    SimpleRandomRotation(),
])

elastic_transform = ElasticDeformation(100, 10.0)


class Dataloader(Dataset):

    def __init__(self, data_info, ids, vol_path, preprocessed_data=False, seg_path=None):
        """
        :param data_info: info of data used (saved as dic in init_data.py)
       :param ids: ids of data to be loaded as numerical list
       :param vol_path: parent directory of data
       :param preprocessed_data: using preprocessed data (True) or pre-processing on the fly (False)
       :param seg_path: parent directory of segmentations (if training data)
        """
        self.data_info = data_info
        self.ids = ids
        self.vol_path = vol_path
        self.seg_path = seg_path
        self.preprocessed_data = preprocessed_data

    def read_volume(self, path, mask=False):
        # reads volume and returns
        data = nib.load(path)
        volume = data.get_fdata()
        # use the T2 image (modility: 0) of this dataset only
        if (self.data_info["name"] == "MD_PROSTATE") and (mask is False) and (self.preprocessed_data is False):
            volume = volume[:, :, :, 0]
        # print("original volume", volume.shape)
        if self.preprocessed_data is False:
            volume = np.moveaxis(volume, -1, 0)
        # header contains info about current pixel_size
        return volume, data.header['pixdim'][1:3], data.affine

    def save_volume(self, path, filename, volume, affine):
        # adapted from: https://github.com/krishnabits001/domain_specific_cl/blob/main/scripts/create_cropped_imgs.py
        if self.data_info["name"] == "ACDC":
            affine[0, 0] = -self.data_info["resolution"][0]
            affine[1, 1] = -self.data_info["resolution"][1]
        elif self.data_info["name"] == "MD_PROSTATE":
            affine[0, 0] = self.data_info["resolution"][0]
            affine[1, 1] = self.data_info["resolution"][1]

        array_vol = nib.Nifti1Image(volume, affine)
        complete_path = os.path.join(str(path), filename)
        nib.save(array_vol, complete_path)
        # print(volume.shape, complete_path)

    def get_filenames(self, idx):
        # addition to filename if loading pre-processed file
        sub = ""
        if self.preprocessed_data:
            sub = "_prep"

        if self.data_info["name"] == "MD_PROSTATE":
            filename = f"prostate_{idx:02d}{sub}.nii.gz"
            filename_seg = filename
        elif self.data_info["name"] == "ACDC":
            sub_folder = f"patient{idx:03d}"
            filenames_dir = next(walk(os.path.join(str(self.vol_path), sub_folder)), (None, None, []))[2]
            # load frame01 object, if not available frame04. Adapted according to implementation of authors
            if f"patient{idx:03d}_frame01.nii.gz" in filenames_dir:
                filename = f"{sub_folder}/patient{idx:03d}_frame01{sub}.nii.gz"
                filename_seg = f"{sub_folder}/patient{idx:03d}_frame01_gt{sub}.nii.gz"
            elif f"patient{idx:03d}_frame04.nii.gz" in filenames_dir:
                filename = f"{sub_folder}/patient{idx:03d}_frame04{sub}.nii.gz"
                filename_seg = f"{sub_folder}/patient{idx:03d}_frame04_gt{sub}.nii.gz"
            else:
                print(f"ERROR: no suitable file found for patient{idx:03d}")

        return filename, filename_seg

    def get_processed_data(self, filename, path, mask):
        # compute path
        full_path = os.path.join(str(path), filename)
        data, resolution, affine = self.read_volume(full_path, mask=mask)

        # pre-process
        if not self.preprocessed_data:
            data = preprocess(data, resolution, self.data_info["resolution"], self.data_info["dimension"],
                              mask=mask, affine=affine)
            self.save_volume(path, f"{filename[:-7]}_prep.nii.gz", data, affine)

        return data

    def get_processed_volume(self, filename):
        return self.get_processed_data(filename, self.vol_path, False)

    def get_processed_seg(self, filename_seg):
        return self.get_processed_data(filename_seg, self.seg_path, True)


class DataloaderRandom(Dataloader):
    def __init__(self, data_info, ids, vol_path, preprocessed_data=False, seg_path=None):
        super().__init__(data_info, ids, vol_path, preprocessed_data, seg_path)
        self.data = self.load_data()
        print(data_info, "final shape", self.data.shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # volume data
        vol = torch.from_numpy(self.data[idx][0, None, :])

        if self.seg_path is None:
            return elastic_transform(vol_transform(vol))
        # segment data
        seg = torch.from_numpy(self.data[idx][1, None, :])

        # make a seed with numpy generator
        seed = np.random.randint(2147483647)

        # fix python and torch random seed (python for custom rotations and torch for torchvision augmentations
        # torch.cuda.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True

        torch.manual_seed(seed)
        np.random.seed(seed)  # Numpy module.
        random.seed(seed)  # Python random module.
        vol = vol_transform(vol)

        torch.manual_seed(seed)
        np.random.seed(seed)  # Numpy module.
        random.seed(seed)  # Python random module.
        vol = elastic_transform(vol)

        torch.manual_seed(seed)
        np.random.seed(seed)  # Numpy module.
        random.seed(seed)  # Python random module.
        seg = seg_transform(seg)

        torch.manual_seed(seed)
        np.random.seed(seed)  # Numpy module.
        random.seed(seed)  # Python random module.
        seg = elastic_transform(seg)

        return vol, one_hot_encoding(seg, self.data_info["num_class"])

    def load_data(self):

        processed_volume = []
        processed_seg = []

        for idx in self.ids:

            filename, filename_seg = self.get_filenames(idx)

            volume = self.get_processed_volume(filename)

            processed_volume.extend(volume)

            if self.seg_path is None:
                continue

            seg = self.get_processed_seg(filename_seg)

            processed_seg.extend(seg)
            # print(idx, len(processed_volume))

        processed_volume_complete = np.array(processed_volume)
        if self.seg_path is None:
            # print("final volume", processed_volume_complete.shape)
            return processed_volume_complete

        processed_seg_complete = np.array(processed_seg)
        # print(processed_volume_complete.shape, processed_seg_complete.shape)
        processed_data_complete = np.stack((processed_volume_complete, processed_seg_complete), axis=0)
        processed_data_complete = np.moveaxis(processed_data_complete, 1, 0)
        # print("final volume", processed_data_complete.shape)
        return processed_data_complete


class DataloaderCustom(Dataloader):
    def __init__(self, data_info, ids, partition, vol_path, preprocessed_data=False, seg_path=None):
        self.pad_frames = 25
        self.padding_list = []
        super().__init__(data_info, ids, vol_path, preprocessed_data, seg_path)
        self.partition = partition
        self.data = self.load_data()
        print(data_info, "final shape", self.data.shape)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        slices = self.data[idx]

        # remove padding
        slices = slices[:, :-self.padding_list[idx]]
        no_all_slices = slices[0].shape[0]

        selected_slices = []
        group_size = no_all_slices // self.partition
        for i in range(self.partition):
            # for last partition, take random slice until the end
            if i == self.partition - 1:
                random_slice_idx = random.randint(i * group_size, no_all_slices - 1)
                # print(f"random value between {i * group_size} and {no_all_slices - 1} is {random_slice_idx}")
            else:
                random_slice_idx = random.randint(i * group_size, (i + 1) * group_size - 1)
                # print(f"random value between {i * group_size} and {(i + 1) * group_size - 1} is {random_slice_idx}")

            random_slice = slices[:, random_slice_idx]
            selected_slices.append(random_slice)

        selected_slices_complete = np.array(selected_slices)

        selected_slices_complete = selected_slices_complete[:, :, None, :, :]

        if self.seg_path is None:
            # print("final slices", selected_slices_complete[:, 0].shape)
            original = torch.from_numpy(selected_slices_complete[:, 0])
            aug1 = elastic_transform(vol_transform(original))
            aug2 = elastic_transform(vol_transform(original))
            return original, aug1, aug2
        # no augmentation here
        return selected_slices_complete[:, 0], one_hot_encoding(selected_slices_complete[:, 1],
                                                                self.data_info["num_class"], True)

    def load_data(self):

        processed_volume = []
        processed_seg = []

        for idx in self.ids:

            filename, filename_seg = self.get_filenames(idx)

            volume = self.get_processed_volume(filename)

            pad = self.pad_frames - volume.shape[0]
            if pad < 0:
                print("ERROR: padding must be increased!")
            self.padding_list.append(pad)
            padding = np.zeros((pad, volume.shape[1], volume.shape[2]))
            volume = np.concatenate((volume, padding))
            # print("volume-shape", volume.shape)
            processed_volume.append(volume)

            if self.seg_path is None:
                continue

            seg = self.get_processed_seg(filename_seg)

            seg = np.concatenate((seg, padding))
            # print("seg-shape", seg.shape)
            processed_seg.append(seg)

        processed_volume_complete = np.array(processed_volume)
        if self.seg_path is None:
            processed_volume_complete = np.expand_dims(processed_volume_complete, axis=0)
            processed_volume_complete = np.moveaxis(processed_volume_complete, 1, 0)
            # print("final volume", processed_volume_complete.shape)
            return processed_volume_complete

        processed_seg_complete = np.array(processed_seg)
        processed_data_complete = np.stack((processed_volume_complete, processed_seg_complete), axis=0)
        processed_data_complete = np.moveaxis(processed_data_complete, 1, 0)
        # print("final volume + seg", processed_data_complete.shape)
        return processed_data_complete


def preprocess(volume, res_old, res_new, dim_new, mask=False, affine=None):
    if not mask:
        volume = minmaxnorm(volume)
    volume = re_sampling(volume, res_old, res_new, dim_new, mask, affine)
    return volume


def minmaxnorm(volume):
    min = np.percentile(volume, 1)
    max = np.percentile(volume, 99)
    return (volume - min) / (max - min)


def re_sampling(volume, res_old, res_new, dim_new, mask, affine):
    re_scale = res_old / res_new

    order = 1
    if mask:
        order = 0

    processed_volume = []

    # if affine:
    # affine_matrix = transform.AffineTransform(affine)

    for slice in volume:
        # transform affine

        # if affine:
        #    processed_slice = transform.warp(slice, affine_matrix)

        # resize image
        processed_slice = transform.rescale(slice, re_scale, order=order, preserve_range=True, mode='constant')

        # crop and padding
        processed_slice = crop_or_pad(processed_slice, dim_new)

        processed_volume.append(processed_slice)

    processed_volume_complete = np.array(processed_volume)

    return processed_volume_complete


def crop_or_pad(slice, dim_new):
    # adapted from https://github.com/krishnabits001/domain_specific_cl/blob/main/dataloaders.py
    x, y = slice.shape

    x_s = (x - dim_new[0]) // 2
    y_s = (y - dim_new[1]) // 2
    x_c = (dim_new[0] - x) // 2
    y_c = (dim_new[1] - y) // 2

    slice_new = np.zeros(dim_new)
    # different scenarios where image is bigger or smaller than desired dimension
    if x > dim_new[0] and y > dim_new[1]:
        slice_new = slice[x_s:x_s + dim_new[0], y_s:y_s + dim_new[1]]
    elif x <= dim_new[0] and y > dim_new[1]:
        slice_new[x_c:x_c + x, :] = slice[:, y_s:y_s + dim_new[1]]
    elif x > dim_new[0] and y <= dim_new[1]:
        slice_new[:, y_c:y_c + y] = slice[x_s:x_s + dim_new[0], :]
    else:
        slice_new[x_c:x_c + x, y_c:y_c + y] = slice[:, :]

    return slice_new


def one_hot_encoding(seg, nb_classes, custom=False):
    if custom:
        seg = torch.nn.functional.one_hot(seg.to(torch.int64), nb_classes).transpose(1, 4).squeeze(-1)
    else:
        seg = torch.nn.functional.one_hot(seg.to(torch.int64), nb_classes).transpose(0, 3).squeeze(-1)
    return seg
