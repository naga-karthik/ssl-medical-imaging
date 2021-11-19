from Dataloader.dataloader import DataloaderCustom, DataloaderRandom
from Dataloader.init_data import md_prostate, acdc

# test_loader = DataloaderCustom(md_prostate, [1, 6], 4, "../Task05_Prostate/images", preprocessed_data=False, seg_path="../Task05_Prostate/labels")

# for x in test_loader:
#    print(x.shape)

# for test_images, test_labels in test_loader:
#    sample_image = test_images[0]
#    sample_label = test_labels[0]
#    print(test_images.shape, test_labels.shape)

test_loader = DataloaderRandom(acdc, [93], "../ACDC_dummy", preprocessed_data=False, seg_path="../ACDC_dummy")

for test_images, test_labels in test_loader:
    sample_image = test_images[0]
    sample_label = test_labels[0]
    # print(test_images.shape, test_images.shape)
    print(test_images[0][90])


test_loader = DataloaderRandom(md_prostate, [1], "../Task05_Prostate_dummy/images", preprocessed_data=True,
                               seg_path="../Task05_Prostate_dummy/labels")

for test_images, test_labels in test_loader:
    sample_image = test_images[0]
    sample_label = test_labels[0]
    # print(test_images.shape, test_images.shape)
