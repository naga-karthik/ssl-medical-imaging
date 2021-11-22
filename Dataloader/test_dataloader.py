from Dataloader.dataloader import DataloaderCustom, DataloaderRandom
from Dataloader.init_data import md_prostate, acdc
import matplotlib.pyplot as plt
import numpy as np
import Dataloader.experiments_paper.data_init_acdc, Dataloader.experiments_paper.data_init_prostate_md

# test_loader = DataloaderCustom(md_prostate, [1, 6], 4, "../Task05_Prostate/images", preprocessed_data=False, seg_path="../Task05_Prostate/labels")

# for x in test_loader:
#    print(x.shape)

# for test_images, test_labels in test_loader:
#    sample_image = test_images[0]
#    sample_label = test_labels[0]
#    print(test_images.shape, test_labels.shape)


no_of_tr_imgs = 'tr1'
# change this to 'c1', 'c2', 'cr3', 'cr4', 'cr5'
comb_of_tr_imgs = 'c1'
# ACDC
img_path = "../ACDC"
seg_path = "../ACDC"
train_ids = Dataloader.experiments_paper.data_init_acdc.train_data(no_of_tr_imgs, comb_of_tr_imgs)
val_ids = Dataloader.experiments_paper.data_init_acdc.val_data(no_of_tr_imgs, comb_of_tr_imgs)
test_ids = Dataloader.experiments_paper.data_init_acdc.test_data()
train_dataset = DataloaderRandom(acdc, train_ids, img_path, preprocessed_data=True, seg_path=seg_path)
val_dataset = DataloaderRandom(acdc, val_ids, img_path, preprocessed_data=True, seg_path=seg_path)
test_dataset = DataloaderRandom(acdc, test_ids, img_path, preprocessed_data=True, seg_path=seg_path)


max = 0
for test_images, test_labels in train_dataset:
    sample_image = test_images[0]
    sample_label = test_labels[0]
    sample_max = len(np.unique(test_labels))
    if sample_max > max:
        max = sample_max

print(max)

# md_prostate
img_path = "../Task05_Prostate/images"
seg_path = "../Task05_Prostate/labels"
train_ids = Dataloader.experiments_paper.data_init_prostate_md.train_data(no_of_tr_imgs, comb_of_tr_imgs)
val_ids = Dataloader.experiments_paper.data_init_prostate_md.val_data(no_of_tr_imgs, comb_of_tr_imgs)
test_ids = Dataloader.experiments_paper.data_init_prostate_md.test_data()
train_dataset = DataloaderRandom(md_prostate, train_ids, img_path, preprocessed_data=True, seg_path=seg_path)
val_dataset = DataloaderRandom(md_prostate, val_ids, img_path, preprocessed_data=True, seg_path=seg_path)
test_dataset = DataloaderRandom(md_prostate, test_ids, img_path, preprocessed_data=True, seg_path=seg_path)

max = 0
for test_images, test_labels in train_dataset:
    sample_image = test_images[0]
    sample_label = test_labels[0]
    sample_max = len(np.unique(test_labels))
    if sample_max > max:
        max = sample_max

print(max)


test_loader = DataloaderCustom(acdc, train_ids, 4, "../ACDC", preprocessed_data=True,
                               seg_path="../ACDC")  # , seg_path="../ACDC")

test = False
if test:
    for test_images in test_loader:
        sample_image = test_images[0]
        # print("test image", test_images.shape)
else:
    for test_images, test_labels in test_loader:
        sample_image = test_images[0]
        sample_label = test_labels[0]
        plt.imshow(np.moveaxis(sample_image, 0, -1))
        # plt.show()
        plt.imshow(np.moveaxis(sample_label, 0, -1))
        # plt.show()
        # print(test_images.shape, test_labels.shape)

# preprocess all data
# test_loader = DataloaderRandom(acdc, range(1,101), "../ACDC", preprocessed_data=False, seg_path="../ACDC")
# test_loader = DataloaderRandom(md_prostate, range(47), "../Task05_Prostate/images", preprocessed_data=False)
# test_loader = DataloaderRandom(md_prostate, [0, 1, 2, 4, 6, 7, 10, 13, 14, 16, 17, 18, 20, 21, 24, 25, 28, 29,
#                                             31, 32, 34, 35, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47],
#                               "../Task05_Prostate/images", preprocessed_data=False, seg_path="../Task05_Prostate"
#                                                                                              "/labels")

# plt.show()
# print(test_images.shape, test_labels.shape)
