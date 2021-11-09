import os
import nibabel as nib
import numpy as np
import cv2
from torch.utils.data import Dataset


class Dataloader(Dataset):
    def __init__(self, data_info, ids, vol_path, seg_path=None):
        """
        :param data_info: info of data used
       :param ids: ids of data to be loaded
       :param vol_path: dir of data
       :param seg_path: path of segmentations (if training data)
        """
        self.data_info = data_info
        self.ids = ids
        self.vol_path = vol_path
        self.seg_path = seg_path

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        if self.data_info["name"] == "MD_PROSTATE":
            filename = f"prostate_{self.ids[idx]:02d}.nii.gz"
        # TODO add specialty other dataset

        # combining path and loading volume
        volume_path = os.path.join(str(self.vol_path), filename)

        volume, resolution = self.read_volume(volume_path)

        # normalization
        volume = minmaxnorm(volume)
        volume = re_sampling(volume, resolution, self.data_info["resolution"], self.data_info["dimension"])

        if self.seg_path is None:
            return volume

        seg_path = os.path.join(str(self.seg_path), filename)
        seg, resolution = self.read_volume(seg_path)

        # normalization
        seg = minmaxnorm(seg)
        seg = re_sampling(seg, resolution, self.data_info["resolution"], self.data_info["dimension"])

        return volume, seg

    def read_volume(self, path):
        # reads volume and returns
        data = nib.load(path)
        # use the T2 image (modility: 0) of this dataset only
        if self.data_info["name"] == "MD_PROSTATE":
            volume = data.get_fdata()[:, :, :, 0]
            volume = np.moveaxis(volume, -1, 0)
            return volume, data.header['pixdim'][1:3]
        # TODO add specialty other dataset
        return


def minmaxnorm(volume):
    min = np.percentile(volume, 1)
    max = np.percentile(volume, 99)
    return (volume - min) / (max - volume)


def re_sampling(volume, res_old, res_new, dim_new):
    print(res_old, res_new)
    re_scale = res_old / res_new

    processed_volume = np.array([])

    for slice in volume:
        dim = tuple(slice.shape * re_scale)
        print(slice.shape, dim)

        # resize image -> some error here with numpy
        resized_slice = cv2.resize(slice, dim, interpolation=cv2.INTER_LINEAR)

        # crop and padding
        processed_slice = crop_or_pad(resized_slice, dim_new)

        processed_volume.append(processed_slice)

    return processed_volume


def crop_or_pad(slice, dim_new):
    # TODO pad when smaller, crop when bigger
    return slice
