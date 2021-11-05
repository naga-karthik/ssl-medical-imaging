import matplotlib.pyplot as plt
import nibabel as nib

"""
Script only used to get an understanding what the data looks like
"""

def show_slices(slices):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")


data_path = "../Task05_Prostate"
img = nib.load(data_path+"/images/prostate_00.nii.gz")

epi_img_data = img.get_fdata()
print(epi_img_data.shape)

slice_0 = epi_img_data[1, :, :, 0]
slice_1 = epi_img_data[:, 1, :, 0]
slice_2 = epi_img_data[:, :, 11, 0]
show_slices([slice_0, slice_1, slice_2])
plt.suptitle("Center slices for EPI image")
plt.show()


data_path = "../Task05_Prostate"
labels = nib.load(data_path+"/labels/prostate_00.nii.gz")
epi_labels = labels.get_fdata()
print(epi_labels.shape)
print(epi_labels[26, 4, 2])
