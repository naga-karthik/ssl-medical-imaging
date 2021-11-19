import os
import random
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset
from os import walk
from skimage import transform


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
        self.data = self.load_data()

    def load_data(self):
        # addition to filename if loading pre-processed file
        sub = ""
        if self.preprocessed_data:
            sub = "_prep"

        processed_volume = []
        processed_seg = []

        for idx in self.ids:
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

            # combining path and loading volume
            volume_path = os.path.join(str(self.vol_path), filename)
            volume, resolution, affine = self.read_volume(volume_path)

            # pre-process
            if not self.preprocessed_data:
                volume = preprocess(volume, resolution, self.data_info["resolution"],
                                    self.data_info["dimension"], mask=False, affine=affine)
                # volume = np.moveaxis(volume, 0, -1)
                print("final volume", volume.shape)
                self.save_volume(self.vol_path, f"{filename[:-7]}_prep.nii.gz", volume, affine)

            if self.__class__.__name__ == "DataloaderRandom":
                processed_volume.extend(volume)
            else:
                # TODO equal partition
                processed_volume.append(volume)

            if self.seg_path is None:
                continue

            seg_path = os.path.join(str(self.seg_path), filename_seg)
            seg, resolution, affine = self.read_volume(seg_path, mask=True)

            # pre-process
            if not self.preprocessed_data:
                seg = preprocess(seg, resolution, self.data_info["resolution"], self.data_info["dimension"],
                                 mask=True, affine=affine)
                self.save_volume(self.seg_path, f"{filename_seg[:-7]}_prep.nii.gz", volume, affine)

            if self.__class__.__name__ == "DataloaderRandom":
                processed_seg.extend(seg)
            else:
                # TODO equal partition
                processed_seg.append(seg)

        processed_volume_complete = np.array(processed_volume)
        processed_seg_complete = np.array(processed_seg)
        processed_data_complete = np.stack((processed_volume_complete, processed_seg_complete), axis=0)
        processed_data_complete = np.moveaxis(processed_data_complete, 1, 0)
        return processed_data_complete

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
        print(volume.shape, complete_path)


class DataloaderRandom(Dataloader):
    def __init__(self, data_info, ids, vol_path, preprocessed_data=False, seg_path=None):
        super().__init__(data_info, ids, vol_path, preprocessed_data, seg_path)
        print(data_info, "final shape", self.data.shape)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self.data[idx]


class DataloaderCustom(Dataloader):
    def __init__(self, data_info, ids, partition, vol_path, preprocessed_data=False, seg_path=None):
        super().__init__(data_info, ids, vol_path, preprocessed_data, seg_path)
        self.partition = partition
        print(data_info, "final shape", self.data.shape)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        slices = self.data[idx]
        # TODO drop empty slices
        randomlist = random.sample(range(0, int(slices.shape[1] / self.partition)), self.partition)
        print(randomlist)
        random_slices = []
        for i in range(self.partition):
            random_slices.append(slices[:, i * self.partition + i * randomlist])

        print("slices", slices.shape, slices.shape[1], int(slices.shape[1] / self.partition))

        # TODO get correct item, 1 item per partition
        return self.data[idx]


def preprocess(volume, res_old, res_new, dim_new, mask=False, affine=None):
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
