import os
import nibabel as nib
from torch.utils.data import Dataset
import numpy as np

class ACDC_dataset(Dataset):
    def __init__(self, frame=None, transform=None):
        assert frame == '01' or frame == '01_gt' or frame == '12' or frame == '12_gt' or frame is None, 'variable frame must be a string of value 01, 01_gt, 12 or 12_gt'
        self.transform = transform
        if frame is None:
            self.frame = '4d.nii.gz'
        else:
            self.frame = f'frame{frame}.nii.gz'

        path = np.empty(0)
        for i in range(1, 101):
            if i - 99 > 0:
                patient_id = f'patient{i}'
            else:
                if i - 10 < 0:
                    patient_id = f'patient00{i}'
                else:
                    patient_id = f'patient0{i}'

            new_path = f'data/dataloader_ACDC/training/{patient_id}/{patient_id}_{self.frame}'
            path = np.append(path, new_path)

        self.img_path = path,

    def __len__(self):
        return len(self.img_path[0])

    def __getitem__(self, img_index):
        return nib.load(self.img_path[0][img_index])

dataset = ACDC_dataset()
print(dataset.img_path)
print(len(dataset))
print(dataset[1])