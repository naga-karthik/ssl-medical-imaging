from Dataloader.dataloader import DataloaderCustom, DataloaderRandom
from Dataloader.init_data import md_prostate, acdc

# test_loader = DataloaderCustom(md_prostate, [1, 6], 4, "../Task05_Prostate/images", preprocessed_data=False, seg_path="../Task05_Prostate/labels")

# for x in test_loader:
#    print(x.shape)

# for test_images, test_labels in test_loader:
#    sample_image = test_images[0]
#    sample_label = test_labels[0]
#    print(test_images.shape, test_labels.shape)

test_loader = DataloaderCustom(acdc, range(1,10), 4, "../ACDC", preprocessed_data=True ) # , seg_path="../ACDC") # , seg_path="../ACDC")

test = True
if test:
    for test_images in test_loader:
        sample_image = test_images[0]
        print("test image", test_images.shape)
else:
    for test_images, test_labels in test_loader:
        sample_image = test_images[0]
        sample_label = test_labels[0]
        print(test_images.shape, test_labels.shape)


# preprocess all data
# test_loader = DataloaderRandom(acdc, range(1,100), "../ACDC", preprocessed_data=False, seg_path="../ACDC")
# test_loader = DataloaderRandom(md_prostate, range(47), "../Task05_Prostate/images", preprocessed_data=False)

test_loader = DataloaderRandom(md_prostate, range(3), "../Task05_Prostate/images", preprocessed_data=False,
                               seg_path="../Task05_Prostate/labels")

for test_images, test_labels in test_loader:
    sample_image = test_images[0]
    sample_label = test_labels[0]
    print(test_images.shape, test_labels.shape)
