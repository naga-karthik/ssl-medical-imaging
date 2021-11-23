# adapted from https://github.com/krishnabits001/domain_specific_cl/blob/main/experiment_init/data_cfg_acdc.py

import sys


def train_data(no_of_tr_imgs, comb_of_tr_imgs):
    # print('train set list')
    if (no_of_tr_imgs == 'tr1' and comb_of_tr_imgs == 'c1'):
        labeled_id_list = [2]
    elif (no_of_tr_imgs == 'tr1' and comb_of_tr_imgs == 'c2'):
        labeled_id_list = [42]
    elif (no_of_tr_imgs == 'tr1' and comb_of_tr_imgs == 'c3'):
        labeled_id_list = [95]
    elif (no_of_tr_imgs == 'tr1' and comb_of_tr_imgs == 'c4'):
        labeled_id_list = [22]
    elif (no_of_tr_imgs == 'tr1' and comb_of_tr_imgs == 'c5'):
        labeled_id_list = [62]
    elif (no_of_tr_imgs == 'tr2' and comb_of_tr_imgs == 'c1'):
        labeled_id_list = [42, 62]
    elif (no_of_tr_imgs == 'tr2' and comb_of_tr_imgs == 'c2'):
        labeled_id_list = [2, 42]
    elif (no_of_tr_imgs == 'tr2' and comb_of_tr_imgs == 'c3'):
        labeled_id_list = [42, 95]
    elif (no_of_tr_imgs == 'tr2' and comb_of_tr_imgs == 'c4'):
        labeled_id_list = [2, 22]
    elif (no_of_tr_imgs == 'tr2' and comb_of_tr_imgs == 'c5'):
        labeled_id_list = [2,95]
    elif (no_of_tr_imgs == 'tr8' and comb_of_tr_imgs == 'c1'):
        labeled_id_list = [2,22,42,62,95,3,23,43]
    elif (no_of_tr_imgs == 'tr8' and comb_of_tr_imgs == 'c2'):
        labeled_id_list = [2,22,42,62,95,63,94,43]
    elif (no_of_tr_imgs == 'tr8' and comb_of_tr_imgs == 'c3'):
        labeled_id_list = [2,22,42,62,95,3,94,23]
    elif (no_of_tr_imgs == 'tr52'):
        # Use this list of subject ids as unlabeled data during pre-training
        labeled_id_list = [1, 2, 3, 4, 5, 6, 12, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 37,38,39,40, 41, 42, 43, 44, 45, 46,
                           57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 77, 78, 79, 80, 72, 81, 82, 83, 84, 85, 86, 97, 98, 99, 100]

    elif (no_of_tr_imgs == 'trall'):
        # Use this list to train the Benchmark/Upperbound
        labeled_id_list = [1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 32,
                           33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 51, 52, 53, 54, 55, 56, 57, 58,
                           59, 60, 61, 62, 63, 64, 65, 66, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85,
                           86, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
    else:
        print('Error! Select valid combination of training images')
        sys.exit()
    return labeled_id_list


def val_data(no_of_tr_imgs, comb_of_tr_imgs):
    # print('val set list')
    if (no_of_tr_imgs == 'tr1' and (comb_of_tr_imgs == 'c1' or comb_of_tr_imgs == 'c5')):
        val_list = [11, 71]
    elif (no_of_tr_imgs == 'tr1' and (comb_of_tr_imgs == 'c2')):
        val_list = [31, 72]
    elif (no_of_tr_imgs == 'tr1' and (comb_of_tr_imgs == 'c3' or comb_of_tr_imgs == 'c4')):
        val_list = [11, 71]
    elif (no_of_tr_imgs == 'tr2' and (comb_of_tr_imgs == 'c2')):
        val_list = [11, 71]
    elif (no_of_tr_imgs == 'tr2' and (comb_of_tr_imgs == 'c1' or comb_of_tr_imgs == 'c3')):
        val_list = [31, 72]
    elif (no_of_tr_imgs == 'tr2' and (comb_of_tr_imgs == 'c4' or comb_of_tr_imgs == 'c5')):
        val_list = [11, 71]
    elif (no_of_tr_imgs == 'tr8' and (comb_of_tr_imgs == 'c1' or comb_of_tr_imgs == 'c3')):
        val_list = [11, 71]
    elif (no_of_tr_imgs == 'tr8' and comb_of_tr_imgs == 'c2'):
        val_list = [31,72]
    elif (no_of_tr_imgs == 'tr52'):
        val_list = [13, 14, 33, 34, 53, 54, 73, 74, 93, 94]
    elif (no_of_tr_imgs == 'trall'):
        val_list = [11, 71]
    return val_list


def test_data():
    # print('test set list')
    test_list = [7, 8, 9, 10, 27, 28, 29, 30, 47, 48, 49, 50, 67, 68, 69, 70, 87, 88, 89, 90]
    return test_list
