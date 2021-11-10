import sys


def train_data(no_of_tr_imgs, comb_of_tr_imgs):
    # print('train set list')
    if (no_of_tr_imgs == 'tr1' and comb_of_tr_imgs == 'c1'):
        labeled_id_list = [1]
    elif (no_of_tr_imgs == 'tr1' and comb_of_tr_imgs == 'c2'):
        labeled_id_list = [2]
    elif (no_of_tr_imgs == 'tr1' and comb_of_tr_imgs == 'c3'):
        labeled_id_list = [6]
    elif (no_of_tr_imgs == 'tr1' and comb_of_tr_imgs == 'c4'):
        labeled_id_list = [0]
    elif (no_of_tr_imgs == 'tr1' and comb_of_tr_imgs == 'c5'):
        labeled_id_list = [4]
    elif (no_of_tr_imgs == 'tr2' and comb_of_tr_imgs == 'c1'):
        labeled_id_list = [0, 1]
    elif (no_of_tr_imgs == 'tr2' and comb_of_tr_imgs == 'c2'):
        labeled_id_list = [21, 24]
    elif (no_of_tr_imgs == 'tr2' and comb_of_tr_imgs == 'c3'):
        labeled_id_list = [1, 2]
    elif (no_of_tr_imgs == 'tr2' and comb_of_tr_imgs == 'c4'):
        labeled_id_list = [4, 10]
    elif (no_of_tr_imgs == 'tr2' and comb_of_tr_imgs == 'c5'):
        labeled_id_list = [0, 7]
    elif (no_of_tr_imgs == 'tr22'):
        # Use this list of subject ids as unlabeled data during pre-training
        labeled_id_list = [0, 1, 2, 4, 6, 7, 10, 13, 14, 16, 17, 18, 20, 21, 24, 25, 28]
    elif (no_of_tr_imgs == 'trall'):
        # Use this list to train the Benchmark/Upperbound
        labeled_id_list = [1, 2, 4, 6, 7, 10, 16, 17, 18, 20, 21, 24, 25, 28]
    elif (no_of_tr_imgs == 'tr8' and comb_of_tr_imgs == 'c1'):
        labeled_id_list = [0, 1, 2, 4, 6, 7, 10, 16]
    elif (no_of_tr_imgs == 'tr8' and comb_of_tr_imgs == 'c2'):
        labeled_id_list = [2, 4, 6, 7, 10, 16, 21, 18]
    elif (no_of_tr_imgs == 'tr8' and comb_of_tr_imgs == 'c3'):
        labeled_id_list = [0, 1, 6, 7, 10, 17, 18, 24]
    else:
        print('Error! Select valid combination of training images')
        sys.exit()
    return labeled_id_list


def val_data(no_of_tr_imgs, comb_of_tr_imgs):
    # print('val set list')
    if (no_of_tr_imgs == 'tr1' and (comb_of_tr_imgs == 'c1' or comb_of_tr_imgs == 'c4')):
        val_list = [13, 14]
    elif (no_of_tr_imgs == 'tr1' and comb_of_tr_imgs == 'c2'):
        val_list = [17, 20]
    elif (no_of_tr_imgs == 'tr1' and (comb_of_tr_imgs == 'c3' or comb_of_tr_imgs == 'c5')):
        val_list = [25, 28]
    elif (no_of_tr_imgs == 'tr2' and (comb_of_tr_imgs == 'c1' or comb_of_tr_imgs == 'c4')):
        val_list = [13, 14]
    elif (no_of_tr_imgs == 'tr2' and comb_of_tr_imgs == 'c2'):
        val_list = [17, 20]
    elif (no_of_tr_imgs == 'tr2' and (comb_of_tr_imgs == 'c3' or comb_of_tr_imgs == 'c5')):
        val_list = [25, 28]
    elif (no_of_tr_imgs == 'tr8' and comb_of_tr_imgs == 'c1'):
        val_list = [13, 14]
    elif (no_of_tr_imgs == 'tr8' and comb_of_tr_imgs == 'c2'):
        val_list = [17, 20]
    elif (no_of_tr_imgs == 'tr8' and comb_of_tr_imgs == 'c3'):
        val_list = [25, 28]
    elif (no_of_tr_imgs == 'tr22'):
        val_list = [13, 14, 20, 28]
    elif (no_of_tr_imgs == 'trall'):
        val_list = [13, 14]
    return val_list


def test_data():
    # print('test set list')
    test_list = [29, 31, 32, 34, 35, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47]
    return test_list
