from Dataloader.dataloader import DataloaderCustom, DataloaderRandom
from Dataloader.init_data import md_prostate, acdc
import matplotlib.pyplot as plt
import numpy as np
import Dataloader.experiments_paper.data_init_acdc, Dataloader.experiments_paper.data_init_prostate_md


def test_custom_seg():
    test_loader = DataloaderCustom(md_prostate, [1, 6], 4, "../Task05_Prostate/images", preprocessed_data=False,
                                   seg_path="../Task05_Prostate/labels")

    for test_images, test_labels in test_loader:
        print(test_images.shape, test_labels.shape)


def test_custom_vol_only():
    test_dataset = DataloaderCustom(md_prostate, [1, 6], 4, "../Task05_Prostate/images", preprocessed_data=False)

    for original, aug1, aug2 in test_dataset:
        print(original.shape, aug1.shape, aug2.shape)


def test_augmentions():
    no_of_tr_imgs = 'tr1'
    # change this to 'c1', 'c2', 'cr3', 'cr4', 'cr5'
    comb_of_tr_imgs = 'c1'
    # ACDC
    img_path = "../ACDC"
    seg_path = "../ACDC"
    train_ids = Dataloader.experiments_paper.data_init_acdc.train_data(no_of_tr_imgs, comb_of_tr_imgs)
    train_dataset = DataloaderRandom(acdc, train_ids, img_path, preprocessed_data=True, seg_path=seg_path)

    for test_images, test_labels in train_dataset:
        fig, axs = plt.subplots(1, 2)
        fig.suptitle('Augementation Original (left) Original with Mask layover (Right)')
        axs[0].imshow(test_images.numpy()[0], cmap='gray')
        axs[1].imshow(test_images.numpy()[0] - test_labels.numpy()[2], cmap='gray')
        fig.show()
        print(test_images.shape, test_labels.shape)

    train_loader = DataloaderCustom(acdc, train_ids, 4, "../ACDC", preprocessed_data=True)

    for original, aug1, aug2 in train_loader:
        for i in range(4):
            fig, axs = plt.subplots(1, 3)
            fig.suptitle('Augmentations')
            axs[0].imshow(original.numpy()[i][0], cmap='gray')
            axs[1].imshow(aug1[i][0], cmap='gray')
            axs[2].imshow(aug2[i][0], cmap='gray')
            fig.show()
        print(original.shape, aug1.shape, aug2.shape)


def test_dataloader_random():
    no_of_tr_imgs = 'tr1'
    # change this to 'c1', 'c2', 'cr3', 'cr4', 'cr5'
    comb_of_tr_imgs = 'c1'
    # md_prostate
    img_path = "../Task05_Prostate/images"
    seg_path = "../Task05_Prostate/labels"
    train_ids = Dataloader.experiments_paper.data_init_prostate_md.train_data(no_of_tr_imgs, comb_of_tr_imgs)
    val_ids = Dataloader.experiments_paper.data_init_prostate_md.val_data(no_of_tr_imgs, comb_of_tr_imgs)
    test_ids = Dataloader.experiments_paper.data_init_prostate_md.test_data()
    train_dataset = DataloaderRandom(md_prostate, train_ids, img_path, preprocessed_data=True, seg_path=seg_path)
    val_dataset = DataloaderRandom(md_prostate, val_ids, img_path, preprocessed_data=True, seg_path=seg_path)
    test_dataset = DataloaderRandom(md_prostate, test_ids, img_path, preprocessed_data=True, seg_path=seg_path)

    for test_images, test_labels in train_dataset:
        print(test_images.shape, test_labels.shape)

    for test_images, test_labels in val_dataset:
        print(test_images.shape, test_labels.shape)

    for test_images, test_labels in test_dataset:
        print(test_images.shape, test_labels.shape)

    # ACDC
    img_path = "../ACDC"
    seg_path = "../ACDC"
    train_ids = Dataloader.experiments_paper.data_init_acdc.train_data(no_of_tr_imgs, comb_of_tr_imgs)
    val_ids = Dataloader.experiments_paper.data_init_acdc.val_data(no_of_tr_imgs, comb_of_tr_imgs)
    test_ids = Dataloader.experiments_paper.data_init_acdc.test_data()
    train_dataset = DataloaderRandom(acdc, train_ids, img_path, preprocessed_data=True, seg_path=seg_path)
    val_dataset = DataloaderRandom(acdc, val_ids, img_path, preprocessed_data=True, seg_path=seg_path)
    test_dataset = DataloaderRandom(acdc, test_ids, img_path, preprocessed_data=True, seg_path=seg_path)

    for test_images, test_labels in train_dataset:
        print(test_images.shape, test_labels.shape)

    for test_images, test_labels in val_dataset:
        print(test_images.shape, test_labels.shape)

    for test_images, test_labels in test_dataset:
        print(test_images.shape, test_labels.shape)

def preprocess_all_data():
    # preprocess all data
    test_loader = DataloaderRandom(acdc, range(1, 101), "../ACDC", preprocessed_data=False, seg_path="../ACDC")
    test_loader = DataloaderRandom(md_prostate, range(47), "../Task05_Prostate/images", preprocessed_data=False)
    test_loader = DataloaderRandom(md_prostate, [0, 1, 2, 4, 6, 7, 10, 13, 14, 16, 17, 18, 20, 21, 24, 25, 28, 29,
                                                 31, 32, 34, 35, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47],
                                   "../Task05_Prostate/images", preprocessed_data=False, seg_path="../Task05_Prostate"
                                                                                                  "/labels")


# code snippet finding out max_no_classes
"""
for test_images, test_labels in test_loader:
    print(test_images.shape, test_labels.shape)
    break
    sample_image = test_images[0]
    sample_label = test_labels[0]
    sample_max = len(np.unique(test_labels))
    if sample_max > max:
        if sample_max < 10:
            # print(sample_max, i)
            max = sample_max
            print(np.unique(test_labels))

    if sample_max > 10:
        print(sample_max, i)
        plt.imshow(np.moveaxis(sample_image, 0, -1))
        plt.show()
        plt.imshow(np.moveaxis(sample_label, 0, -1))
        plt.show()

    i += 1

print(max)
"""

test_custom_vol_only()