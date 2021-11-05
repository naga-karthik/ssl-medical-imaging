import os
import nibabel as nib
import numpy as np
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
            # TODO fix
            filename = f"prostate_{self.ids[idx]:02}.nii.gz"
        volume_path = os.path.join(str(self.vol_path), filename)
        volume = self.read_volume(volume_path)
        volume = minmaxnorm(volume)
        print(volume.shape)
        images = slice_volume(volume)

        if self.seg_path is None:
            return images

        seg_path = os.path.join(str(self.seg_path), filename)
        seg = self.read_volume(seg_path)

        return images, seg

    def read_volume(self, path):
        # reads volume and returns
        volume = nib.load(path)
        # use the T2 image (modility: 0) of this dataset only
        if self.data_name == "MD_PROSTATE":
            return volume.get_fdata()[0]
        return volume.get_fdata()


def minmaxnorm(volume):
    min = np.percentile(volume, 1)
    max = np.percentile(volume, 99)
    return (volume - min) / (max - volume)

# this is probably just a numpy reshape
def slice_volume(volume):
    return volume
