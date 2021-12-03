from Dataset.dataset import DatasetGDMinus, DatasetRandom, DatasetGR
from Dataset.init_data import md_prostate, acdc
import matplotlib.pyplot as plt
import numpy as np
import torch
import Dataset.experiments_paper.data_init_acdc, Dataset.experiments_paper.data_init_prostate_md
from PIL import Image


def test_custom_seg():
    test_loader = DatasetGDMinus(md_prostate, [1, 6], 4, "../Task05_Prostate/images", preprocessed_data=False,
                                    seg_path="../Task05_Prostate/labels")

    for test_images, test_labels in test_loader:
        print(test_images.shape, test_labels.shape)


def test_custom_vol_only():
    test_dataset = DatasetGDMinus(md_prostate, [1, 6], 4, "../Task05_Prostate/images", preprocessed_data=False)

    for original, aug1, aug2 in test_dataset:
        print(original.shape, aug1.shape, aug2.shape)


def test_augmentions():
    no_of_tr_imgs = 'tr1'
    # change this to 'c1', 'c2', 'cr3', 'cr4', 'cr5'
    comb_of_tr_imgs = 'c1'
    # ACDC
    img_path = "../ACDC"
    seg_path = "../ACDC"
    train_ids = Dataset.experiments_paper.data_init_acdc.train_data(no_of_tr_imgs, comb_of_tr_imgs)
    train_dataset = DatasetRandom(acdc, train_ids, img_path, preprocessed_data=True, seg_path=seg_path,
                                     augmentation=True)

    for test_images, test_labels in train_dataset:
        print(test_images.shape, test_labels.shape)
        fig, axs = plt.subplots(1, 2)
        fig.suptitle('Augementation Original (left) Original with Mask layover (Right)')
        axs[0].imshow(test_images.numpy()[0], cmap='gray')
        axs[1].imshow(test_images.numpy()[0] - test_labels.numpy()[0], cmap='gray')
        fig.show()

    train_loader = DatasetGDMinus(acdc, train_ids, 4, "../ACDC", preprocessed_data=True)

    for original, aug1, aug2 in train_loader:
        for i in range(4):
            fig, axs = plt.subplots(1, 3)
            fig.suptitle('Augmentations')
            # fft_p = (original.numpy()[0][i]).astype(np.uint8)
            fft_p = original.numpy()[i][0]
            im = Image.fromarray(fft_p * 255.0)
            im = im.convert('L')
            im.save(f"img_{i}.png")
            print(original.numpy()[i].shape)
            axs[0].imshow(original.numpy()[i, 0], cmap='gray')
            axs[1].imshow(aug1.numpy()[i, 0], cmap='gray')
            axs[2].imshow(aug2.numpy()[i, 0], cmap='gray')
            fig.show()
        print(original.shape, aug1.shape, aug2.shape)


def test_dataloader_random():
    no_of_tr_imgs = 'tr1'
    # change this to 'c1', 'c2', 'cr3', 'cr4', 'cr5'
    comb_of_tr_imgs = 'c1'
    # md_prostate
    img_path = "../Task05_Prostate/images"
    seg_path = "../Task05_Prostate/labels"
    train_ids = Dataset.experiments_paper.data_init_prostate_md.train_data(no_of_tr_imgs, comb_of_tr_imgs)
    val_ids = Dataset.experiments_paper.data_init_prostate_md.val_data(no_of_tr_imgs, comb_of_tr_imgs)
    test_ids = Dataset.experiments_paper.data_init_prostate_md.test_data()
    train_dataset = DatasetRandom(md_prostate, train_ids, img_path, preprocessed_data=True, seg_path=seg_path,
                                     augmentation=True)
    val_dataset = DatasetRandom(md_prostate, val_ids, img_path, preprocessed_data=True, seg_path=seg_path)
    test_dataset = DatasetRandom(md_prostate, test_ids, img_path, preprocessed_data=True)

    for test_images, test_labels in train_dataset:
        print(test_images.shape, test_labels.shape, torch.min(test_labels), torch.max(test_labels))

    for test_images, test_labels in val_dataset:
        print(test_images.shape, test_labels.shape, torch.min(test_labels), torch.max(test_labels))

    for test_images in test_dataset:
        print(test_images.shape)

    # ACDC
    img_path = "../ACDC"
    seg_path = "../ACDC"
    train_ids = Dataset.experiments_paper.data_init_acdc.train_data(no_of_tr_imgs, comb_of_tr_imgs)
    val_ids = Dataset.experiments_paper.data_init_acdc.val_data(no_of_tr_imgs, comb_of_tr_imgs)
    test_ids = Dataset.experiments_paper.data_init_acdc.test_data()
    train_dataset = DatasetRandom(acdc, train_ids, img_path, preprocessed_data=True, seg_path=seg_path,
                                     augmentation=True)
    val_dataset = DatasetRandom(acdc, val_ids, img_path, preprocessed_data=True, seg_path=seg_path)
    test_dataset = DatasetRandom(acdc, test_ids, img_path, preprocessed_data=True)

    for test_images, test_labels in train_dataset:
        print(test_images.shape, test_labels.shape, torch.min(test_labels), torch.max(test_labels))

    for test_images, test_labels in val_dataset:
        print(test_images.shape, test_labels.shape, torch.min(test_labels), torch.max(test_labels))

    for test_images in test_dataset:
        print(test_images.shape)


def test_dataloaderGR():
    no_of_tr_imgs = 'tr1'
    # change this to 'c1', 'c2', 'cr3', 'cr4', 'cr5'
    comb_of_tr_imgs = 'c1'
    # md_prostate
    img_path = "../Task05_Prostate/images"
    train_ids = Dataset.experiments_paper.data_init_prostate_md.train_data(no_of_tr_imgs, comb_of_tr_imgs)
    train_dataset = DatasetGR(md_prostate, train_ids, img_path, preprocessed_data=True)

    for aug1, aug2 in train_dataset:
        print(aug1.shape, aug2.shape)

    # ACDC
    img_path = "../ACDC"
    train_ids = Dataset.experiments_paper.data_init_acdc.train_data(no_of_tr_imgs, comb_of_tr_imgs)
    train_dataset = DatasetGR(acdc, train_ids, img_path, preprocessed_data=True)

    for aug1, aug2 in train_dataset:
        print(aug1.shape, aug2.shape)

def test_dataloaderGDMinus():
    no_of_tr_imgs = 'tr1'
    # change this to 'c1', 'c2', 'cr3', 'cr4', 'cr5'
    comb_of_tr_imgs = 'c1'
    # md_prostate
    img_path = "../Task05_Prostate/images"
    train_ids = Dataset.experiments_paper.data_init_prostate_md.train_data(no_of_tr_imgs, comb_of_tr_imgs)
    train_dataset = DatasetGDMinus(md_prostate, train_ids, 4, img_path, preprocessed_data=True)

    for original, aug1, aug2 in train_dataset:
        print(original.shape, aug1.shape, aug2.shape)

    # ACDC
    img_path = "../ACDC"
    train_ids = Dataset.experiments_paper.data_init_acdc.train_data(no_of_tr_imgs, comb_of_tr_imgs)
    train_dataset = DatasetGDMinus(acdc, train_ids, 4, img_path, preprocessed_data=True)

    for original, aug1, aug2 in train_dataset:
        print(original.shape, aug1.shape, aug2.shape)


def preprocess_all_data():
    # preprocess all data
    dataset_acdc = DatasetRandom(acdc, range(1, 101), "../ACDC", preprocessed_data=False, seg_path="../ACDC")
    DatasetRandom(md_prostate, range(47), "../Task05_Prostate/images", preprocessed_data=False)
    dataset_mdprostate = DatasetRandom(md_prostate,
                                          [0, 1, 2, 4, 6, 7, 10, 13, 14, 16, 17, 18, 20, 21, 24, 25, 28, 29,
                                           31, 32, 34, 35, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47],
                                          "../Task05_Prostate/images", preprocessed_data=False,
                                          seg_path="../Task05_Prostate"
                                                   "/labels")
    for test_images, test_labels in dataset_acdc:
        print(test_images.shape, test_labels.shape, torch.min(test_labels), torch.max(test_labels))

    for test_images, test_labels in dataset_mdprostate:
        print(test_images.shape, test_labels.shape, torch.min(test_labels), torch.max(test_labels))


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

# test_custom_vol_only()

# test_augmentions()
# test_dataloader_random()
# test_augmentions()
# preprocess_all_data()
test_dataloader_random()
# test_dataloaderGR()
# test_dataloaderGDMinus()
