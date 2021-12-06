import torch
from torch._C import device
import torch.nn.functional as F
from torch import Tensor
import seg_models

# Taken from: https://github.com/milesial/Pytorch-UNet/blob/master/utils/dice_score.py
def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter
        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]

'''
Inputs:
    loss_type: 0 for dice loss (default), 1 for global loss, 2 for local loss
    encoder_strategy: 0 for random (default), 1 for Gr, 2 dor Gd-, 3 for Gd
    decoder_strategy: 0 for random (default), 1 for Lr, 2 for Ld
'''
class Loss:
    def __init__(self, loss_type=0, encoder_strategy=0, decoder_strategy=0, device='cpu'):
        self.decoder_strategy = decoder_strategy
        self.encoder_strategy = encoder_strategy
        self.loss_type = loss_type
        self.tau = 0.1
        self.smooth = 0.001     # smoothing factor for the dice loss
        self.device = device

    def one_hot(self, arr, num_classes):
        # converting arr into a LongTensor so that it can be used as indices
        arr = arr.squeeze()
        one_hot_encoded = torch.eye(num_classes)[arr]   # shape: [batch_size, 192, 192, num_classes]
        return one_hot_encoded.permute(0, 3, 1, 2)      # shape: [batch_size, num_classes, 192, 192]

    def dice_loss_v2(self, input: Tensor, target: Tensor, multiclass: bool = False):
        # Dice loss (objective to minimize) between 0 and 1
        assert input.size() == target.size()
        fn = multiclass_dice_coeff if multiclass else dice_coeff
        return 1 - fn(input, target, reduce_batch_first=True)

    def dice_loss(self, prediction, target, test=False):
        # should output number of unique classes
        # classes = torch.unique(target)      # causing issues, num_classes is wrong
        # print(f"\nclasses shape: {classes.shape}")

        c = prediction.shape[1]  # classes.shape[0]
        tflat = target.view(-1)
        tflat_one_hot = self.one_hot(tflat, c)
        pflat_softmax = prediction.view(-1, c)

        intersection_of_label_with_image = torch.sum(torch.mul(tflat_one_hot, pflat_softmax), dim=1)
        total_sum = torch.sum((pflat_softmax + tflat_one_hot), dim=1)
        dices = (2.0 * intersection_of_label_with_image + self.smooth) / (total_sum + self.smooth)

        if test:
            return torch.mean(dices)
        else:
            return 1.0 - torch.mean(dices)

    # cosine similarity
    def cos_sim1(self, vect1, vect2):
        vect1_norm = F.normalize(vect1, dim=-1, p=2)
        vect2_norm = torch.transpose(F.normalize(vect2, dim=-1, p=2), 0, -1)
        return torch.log(torch.matmul(vect1_norm, vect2_norm)) / 0.1  # scale factor 0.1 applied in online repo
    
    def cos_sim(self, all_latents):
        r = all_latents.size(dim=0)
        sim_arr = torch.zeros(r, r)
        for i in range(r):
            for j in range(r):
                x = self.cos_sim1(all_latents[i, :], all_latents[j, :]) # using our original cos similarity for 2 latents to compute the entire matrix of cos similarities
                print(f'vecti: {all_latents[i, :10]} vectj: {all_latents[j, :10]}, x: {x}')
                
                sim_arr[i, j] = x
        return sim_arr

    
    '''def create_pos_set(self, aug1, aug2, unaug):
        n = aug1.size(dim=0)
        c = aug1.size(dim=1)

        # array pos contains all positive pairs for each strategy.  
        # array aug_types contains the augmentation type of each positive pair: 0=none, 1=aug1, 2 = aug2
        if self.encoder_strategy == 1:
           pos = torch.zeros(n, 2, c)
           aug_type = torch.zeros(n, 2)
        if self.encoder_strategy == 2:
            pos = torch.zeros(int(n*3), 2, c)
            aug_type = torch.zeros(int(n*3), 2)
        if self.encoder_strategy == 3:
            pos = torch.zeros(int(n*4), 2, c)
            aug_type = torch.zeros(int(n*4), 2)


        for i in range(n):
            # loss type Gr only uses aug1 and aug2, using notation from the paper: pos pairs are (x_i_bar, x_i_hat)
            # where the "i" indicates the augmented images come from the same original image "x_i"
            if self.encoder_strategy == 1:
                pairs = torch.zeros(1, 2, c)
                pairs[0, 0, :] = aug1[i, :]
                pairs[0, 1, :] = aug2[i, :]
                aug_type[i, :] = torch.tensor([1., 2.])
                pos[i, :, :] = pairs

            # loss type Gd- uses aug1 and aug2 and noaug
            # using notation from the paper: pos pairs are (x_i_bar, x_i_hat), (x_i, x_i_hat), (x_i, x_i_bar)
            # adds each of the 3 pos pairs at each iteration
            if self.encoder_strategy == 2:
                pairs = torch.zeros(3, 2, 128)
                pairs[1, 1, :] = aug1[i, :]
                pairs[1, 2, :] = aug2[i, :]
                aug_type[int(i*3), :] = torch.tensor([1., 2.])
                pairs[2, 1, :] = unaug[i, :]
                pairs[2, 2, :] = aug1[i, :]
                aug_type[int(i*3) + 1., :] = torch.tensor([0., 1.])
                pairs[3, 1, :] = unaug[i, :]
                pairs[3, 2, :] = aug2[i, :]
                aug_type[int(i*3) + 2., :] = torch.tensor([0., 2.])
                pos[int(i*3):int(i*3) + 3., :, :] = pairs

            # loss type Gd uses aug1 and aug2 and noaug
            # using notation from the paper: pos pairs are (x_i_bar, x_i_hat), (x_i, x_i_hat), (x_i, x_i_bar), (x_i, x_j)
            # x_i, x_i_hat and x_i_bar are all from the same source image, x_j is a different source image from the same partition as x_i
            if self.encoder_strategy == 3:
                # TODO: figure out how to get 2nd image for pos pairs (need partition info)
                pairs = torch.zeros(4, 2, c)
        
        # returns tensor of positive sets and a tensor indicating the augmentation type of each image
        return pos, aug_type

    # create negative set consisting of images that are dissimilar to x_i and its transformed versions.
    # This set may include all images other than x_i, including their possible transformations.
    def create_neg_set(self, aug1, aug2, unaug, pos_set_idx):
        n = aug1.size(dim=0)
        c = aug1.size(dim=1)

        if self.encoder_strategy == 1:
            neg = torch.zeros(int((2*n)-2), c)
        if self.encoder_strategy == 2:
            neg = torch.zeros(int((3*n)-3), c) # wrong initialization needs update to remove any images from that same partition as well
        if self.encoder_strategy == 3:
            neg = torch.zeros(int((4*n)-4), c) # wrong initialization needs update to remove any images from that same partition as well
        
        # returns a tensor with x_i and any augmentations of x_i removed
        if self.encoder_strategy == 1:
            skip = False # flag to modify idx after skip
            for i in range(n):
                if i == pos_set_idx:
                    skip = True
                else:
                    if skip:
                        j = i - 1
                    else:
                        j = i
                    neg[int(2*j), :] = aug1[j, :]
                    neg[int((2*j)+1), :] = aug2[j, :]
        else:
            # TODO: implement neg pair sorting, need partition info
            pass
        # TODO: maybe figure out how to shuffle neg before returning?        
        return neg'''

    # implementation of eq 1 from the paper
    '''def individual_global_loss(self, latent1, latent2, neg_set):
        n = neg_set.size(0)
        # numerator term in eq 1
        num = torch.exp(self.cos_sim(latent1, latent2) / self.tau)
        # denominator term in eq 1
        denom = 0
        for i in range(n):
            denom += torch.exp(self.cos_sim(latent1, neg_set[i, :]))
        denom += num
        # return eq 1
        return -torch.log(torch.div(num, denom))'''

    def individual_global_loss(self, latent1_idx, latent2_idx, cos_sim_arr):
        c = cos_sim_arr.size(dim=1) # num cols
        # numerator term in eq 1
        num = torch.exp(cos_sim_arr[latent1_idx, latent2_idx] / self.tau)

        # get all cols in row latent1_idx except for col latent1_idx
        col_booleans = torch.ones(c, dtype=torch.bool)
        col_booleans[latent1_idx] = False
        # cos sims to sum for denominator
        denom_terms = cos_sim_arr[latent1_idx, col_booleans]
        #print(cos_sim_arr[latent1_idx, :], cos_sim_arr[latent1_idx, col_booleans])
        # denominator term in eq 1
        denom = torch.sum(
            torch.exp(denom_terms / self.tau)
        )
        loss = -torch.log(torch.div(num, denom))
        #print(f'NUM: {num}, DENOM: {denom}, LOSS: {loss}')

        return loss.squeeze(0)

    # implementation of eq 2 from the paper
    '''def global_loss(self, aug1, aug2, unaug):
        loss = 0
        pos_set, aug_type = self.create_pos_set(aug1, aug2, unaug)
        n = pos_set.size(dim=0)
        
        # loop through all images in the pos set
        for i in range(n):
            # create neg set for the current pos set
            neg_set = self.create_neg_set(aug1, aug2, unaug, i)
            
            # pass pos_im1, pos_im2 and neg set to individual loss function<-- image order matters!
            l1 = self.individual_global_loss(pos_set[i, 0, :], pos_set[i, 1, :], neg_set)
            # pass pos_im2, pos_im1 and neg set to individual loss function <-- image order matters!
            l2 = self.individual_global_loss(pos_set[i, 1, :], pos_set[i, 0, :], neg_set)
            
            loss += l1 + l2

        return loss / n'''

    def global_loss(self, aug1, aug2, unaug):
        loss = 0
        n = aug1.size(dim=0)

        # combine latents from each array into 1
        combine = torch.cat((aug1, aug2), dim=0) # gives a 2*n x 128 arr of the stacked latents

        # calculate cos similarity between each image pair
        cos_sim_arr = self.cos_sim(combine)
        print(cos_sim_arr)

        for i in range(n):
            latent1_idx = int(i)
            latent2_idx = int(i + n)
            loss += self.individual_global_loss(latent1_idx, latent2_idx, cos_sim_arr) + self.individual_global_loss(latent2_idx, latent1_idx, cos_sim_arr)

    # this should be the only function you ever call, everything else is called as a chain reaction from here depending 
    # on the Loss object you created, loss type 0 is dice, loss type 1 is global
    def compute(self, aug1, aug2=None, unaug=None, target=None, multiclass=False):
        aug1 = aug1.to(self.device)
        aug1 = F.normalize(aug1, dim=1) # normalize latents
        if aug2 != None:
            aug2 = aug2.to(self.device)
            aug2 = F.normalize(aug2, dim=1) # normalize latents
        if unaug != None:
            unaug = unaug.to(self.device)
            unaug = F.normalize(unaug, dim=1)  # normalize latents
        if self.loss_type == 0:
            target = target.to(self.device)
            return self.dice_loss_v2(aug1, target, multiclass)    # the new, "working" dice loss
        if self.loss_type == 1:
            return self.global_loss(aug1, aug2, unaug)
