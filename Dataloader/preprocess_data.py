# script to crop the images into the target resolution and save them,
# adapted from: https://github.com/krishnabits001/domain_specific_cl/blob/main/scripts/create_cropped_imgs.py
import numpy as np
import pathlib

import nibabel as nib
from os import path
from Dataloader.init_data import md_prostate, acdc

# TODO
from Dataloader.init_data import md_prostate, acdc

img_path_md_prostate = "../Task05_Prostate/images"
seg_path_md_prostate = "../Task05_Prostate/labels"
dataset = md_prostate

######################################
# class loaders
# ####################################
#  load dataloader object
if dataset["name"] == "MD_PROSTATE":
    ids =


for idx in ids:
    if data_info["name"] == "MD_PROSTATE":
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
    volume, resolution, affine = read_volume(volume_path)

    # pre-process
    if not self.preprocessed_data:
        volume = preprocess(volume, resolution, self.data_info["resolution"],
                            self.data_info["dimension"], mask=False, affine=affine)

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

    if self.__class__.__name__ == "DataloaderRandom":
        processed_seg.extend(seg)
    else:
        # TODO equal partition
        processed_seg.append(seg)

dt = dataloaderObj(cfg)

if parse_config.dataset == 'acdc':
    # print('set acdc orig img dataloader handle')
    orig_img_dt = dt.load_acdc_imgs
    start_id, end_id = 1, 101
elif parse_config.dataset == 'prostate_md':
    # print('set prostate_md orig img dataloader handle')
    orig_img_dt = dt.load_prostate_imgs_md
    start_id, end_id = 0, 48

# For loop to go over all available images
for index in range(start_id, end_id):
    if (index < 10):
        test_id = '00' + str(index)
    elif (index < 100):
        test_id = '0' + str(index)
    else:
        test_id = str(index)
    test_id_l = [test_id]

    if parse_config.dataset == 'acdc':
        file_path = str(cfg.data_path_tr) + str(test_id) + '/patient' + str(test_id) + '_frame01.nii.gz'
        mask_path = str(cfg.data_path_tr) + str(test_id) + '/patient' + str(test_id) + '_frame01_gt.nii.gz'
    elif parse_config.dataset == 'prostate_md':
        file_path = str(cfg.data_path_tr) + str(test_id) + '/img.nii.gz'
        mask_path = str(cfg.data_path_tr) + str(test_id) + '/mask.nii.gz'

    # check if image file exists
    if (path.exists(file_path)):
        print('crop', test_id)
    else:
        print('continue', test_id)
        continue

    # check if mask exists
    if path.exists(mask_path):
        # Load the image &/mask
        img_sys, label_sys, pixel_size, affine_tst = orig_img_dt(test_id_l, ret_affine=1, label_present=1)
        # Crop the loaded image &/mask to target resolution
        cropped_img_sys, cropped_mask_sys = dt.preprocess_data(img_sys, label_sys, pixel_size)
    else:
        # Load the image &/mask
        img_sys, pixel_size, affine_tst = orig_img_dt(test_id_l, ret_affine=1, label_present=0)
        # dummy mask with zeros
        label_sys = np.zeros_like(img_sys)
        # Crop the loaded image &/mask to target resolution
        cropped_img_sys = dt.preprocess_data(img_sys, label_sys, pixel_size, label_present=0)

    # output directory to save cropped image &/mask
    save_dir_tmp = str(cfg.data_path_tr_cropped) + str(test_id) + '/'
    pathlib.Path(save_dir_tmp).mkdir(parents=True, exist_ok=True)

    if (parse_config.dataset == 'acdc'):
        affine_tst[0, 0] = -cfg.target_resolution[0]
        affine_tst[1, 1] = -cfg.target_resolution[1]
    elif (parse_config.dataset == 'prostate_md'):
        affine_tst[0, 0] = cfg.target_resolution[0]
        affine_tst[1, 1] = cfg.target_resolution[1]

    # Save the cropped image &/mask
    array_img = nib.Nifti1Image(cropped_img_sys, affine_tst)
    pred_filename = str(save_dir_tmp) + 'img_cropped.nii.gz'
    nib.save(array_img, pred_filename)
    if path.exists(mask_path):
        array_mask = nib.Nifti1Image(cropped_mask_sys.astype(np.int16), affine_tst)
        pred_filename = str(save_dir_tmp) + 'mask_cropped.nii.gz'
        nib.save(array_mask, pred_filename)