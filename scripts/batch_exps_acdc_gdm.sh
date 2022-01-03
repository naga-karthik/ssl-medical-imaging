#!/bin/bash

dataset='ACDC'
imgs_path_acdc='/home/ssl_project/datasets/ACDC' 
segs_path_acdc='/home/ssl_project/datasets/ACDC' 
# cv_folds='c1 c2 c3 c4 c5'
cv_folds='c3 c2 c1'
ft_vols='tr2 tr8'

# do it for tr1 separately
for cvfs in $cv_folds; do
    CUDA_VISIBLE_DEVICES=2 python finetune.py -ptr -data $dataset --img_path $imgs_path_acdc --seg_path $segs_path_acdc -st GD- -nti tr1 -cti $cvfs -bs 8
done    

# do it for tr8 and tr2 in a loop
for ftvs in $ft_vols; do
    for cvfs in $cv_folds; do
        CUDA_VISIBLE_DEVICES=2 python finetune.py -ptr -data $dataset --img_path $imgs_path_acdc --seg_path $segs_path_acdc -st GD- -nti $ftvs -cti $cvfs -bs 12
    done    
done
