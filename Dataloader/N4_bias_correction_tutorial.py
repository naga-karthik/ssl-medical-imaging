#!/usr/bin/env python

from __future__ import print_function

import SimpleITK as sitk
import sys
import os

in_file_name='../Task05_Prostate/images/prostate_00.nii.gz'
out_file_name='../Task05_Prostate/images/prostate_00_bias_corr.nii.gz'
# in_file_name= "../ACDC/patient001/patient001_frame01.nii.gz"
# out_file_name="../ACDC/patient001/patient001_frame01_bias_corr.nii.gz"
threshold_value = 0.001
numberFittingLevels = 4
n_iters = 50

inputImage = sitk.ReadImage(in_file_name, sitk.sitkFloat32)
image = inputImage

# maskImage = sitk.ReadImage(sys.argv[4], sitk.sitkUint8)
maskImage = sitk.OtsuThreshold(inputImage, 0, 1, 200)

corrector = sitk.N4BiasFieldCorrectionImageFilter()

corrector.SetMaximumNumberOfIterations([n_iters] * numberFittingLevels)

corrected_image = corrector.Execute(image, maskImage)

log_bias_field = corrector.GetLogBiasFieldAsImage(inputImage)

bias_field = inputImage / sitk.Exp( log_bias_field )

sitk.WriteImage(corrected_image, out_file_name)

if ("SITK_NOSHOW" not in os.environ):
    sitk.Show(corrected_image, "N4 Corrected")