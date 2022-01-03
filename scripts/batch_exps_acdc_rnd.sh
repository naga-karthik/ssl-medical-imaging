#!/bin/bash

dataset='ACDC' 
imgs_path_acdc='/home/ssl_project/datasets/ACDC' 
segs_path_acdc='/home/ssl_project/datasets/ACDC' 
cv_folds='c1 c2 c3'
ft_vols='tr2 tr8'

# for i in ${!imgs_path[@]}; do
#     echo ${imgs_path[$i]}
# done

# do it for tr1 separately
for cvfs in $cv_folds; do
    CUDA_VISIBLE_DEVICES=3 python supervised_train.py -data $dataset --img_path $imgs_path_acdc --seg_path $segs_path_acdc -nti tr1 -cti $cvfs -bs 8
done    

# do it for tr8 and tr2 in a loop
for ftvs in $ft_vols; do
    for cvfs in $cv_folds; do
        CUDA_VISIBLE_DEVICES=3 python supervised_train.py -data $dataset --img_path $imgs_path_acdc --seg_path $segs_path_acdc -nti $ftvs -cti $cvfs -bs 12
    done    
done


