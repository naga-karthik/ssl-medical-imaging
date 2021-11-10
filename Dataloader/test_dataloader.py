from Dataloader.dataloader import Dataloader
from Dataloader.init_data import md_prostate, acdc
import numpy as np

test_loader = Dataloader(md_prostate, [0, 1], "../Task05_Prostate/images", seg_path="../Task05_Prostate/labels")

for test_images, test_labels in test_loader:
    sample_image = test_images[0]
    sample_label = test_labels[0]

test_loader = Dataloader(acdc, [90], "../ACDC", seg_path="../ACDC")

for test_images, test_labels in test_loader:
    sample_image = test_images[0]
    sample_label = test_labels[0]
