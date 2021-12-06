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

    # old cosine similarity, the results were better using the pytorch version but I'm keeping it to troubleshoot
    def cos_sim_old(self, vect1, vect2):
        vect1_norm = F.normalize(vect1, dim=-1, p=2)
        vect2_norm = torch.transpose(F.normalize(vect2, dim=-1, p=2), 0, -1)
        return torch.matmul(vect1_norm, vect2_norm)

    # cosine similarity using pytorch cosine similarity
    def cos_sim(self, all_latents):
        r = all_latents.size(dim=0)
        sim_arr = torch.zeros(r, r)
        for i in range(r):
            for j in range(r):
                #x = self.cos_sim1(all_latents[i, :], all_latents[j, :])
                x = F.cosine_similarity(all_latents[i, :], all_latents[j, :], 0)
                sim_arr[i, j] = x
        return sim_arr

    def individual_global_loss(self, latent1_idx, latent2_idx, cos_sim_arr):
        c = cos_sim_arr.size(dim=1) # num cols
        # numerator term in eq 1
        num = torch.exp(cos_sim_arr[latent1_idx, latent2_idx] / self.tau)

        # get all cols in row latent1_idx except for col latent1_idx
        col_booleans = torch.ones(c, dtype=torch.bool)
        col_booleans[latent1_idx] = False
        # cos sims to sum for denominator
        denom_terms = cos_sim_arr[latent1_idx, col_booleans]
        # denominator term in eq 1
        denom = torch.sum(
            torch.exp(denom_terms / self.tau)
        )
        loss = -torch.log(torch.div(num, denom))

        return loss

    # implementation of eq 2 from the paper

    def global_loss(self, aug1, aug2, unaug):
        loss = 0
        n = aug1.size(dim=0)

        # set the number of positive pairs for each strategy
        if self.encoder_strategy == 1:
            num_pos = n
        if self.encoder_strategy == 2:
            num_pos = n * 3
        if self.encoder_strategy == 3:
            num_pos = n * 4

        # combine latents from each array into 1
        combine = torch.cat((aug1, aug2), dim=0) # gives a 2*n x 128 arr of the stacked latents

        # calculate cos similarity between each image pair
        cos_sim_arr = self.cos_sim(combine)

        for i in range(n):
            latent1_idx = int(i)
            latent2_idx = int(i + n)
            loss += self.individual_global_loss(latent1_idx, latent2_idx, cos_sim_arr) + self.individual_global_loss(latent2_idx, latent1_idx, cos_sim_arr)
        
        return loss / num_pos
        
    # this should be the only function you ever call, everything else is called as a chain reaction from here depending 
    # on the Loss object you created, loss type 0 is dice, loss type 1 is global
    def compute(self, aug1, aug2=None, unaug=None, target=None, multiclass=False):
        aug1 = aug1.to(self.device)
        if aug2 != None:
            aug2 = aug2.to(self.device)
        if unaug != None:
            unaug = unaug.to(self.device)
        if self.loss_type == 0:
            target = target.to(self.device)
            return self.dice_loss_v2(aug1, target, multiclass)    # the new, "working" dice loss
        if self.loss_type == 1:
            return self.global_loss(aug1, aug2, unaug)
