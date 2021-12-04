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
    def cos_sim(self, vect1, vect2):
        vect1_norm = F.normalize(vect1, dim=-1, p=2)
        vect2_norm = torch.transpose(F.normalize(vect2, dim=-1, p=2), -1, -2)
        return torch.matmul(vect1_norm, vect2_norm)


    # array aug_types contains on entry per sample specifying the augmentation type: 0=none, 1=aug1, 2 = aug2
    def create_pos_set(self, aug1, aug2, unaug):
        n = aug1.size(dim=0)
        c = aug1.size(dim=1)

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
            if self.encoder_strategy == 1:
                pairs = torch.zeros(1, 2, c)
                pairs[1, 1, :] = aug1[i, :]
                pairs[1, 2, :] = aug2[i, :]
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

    def create_neg_set(self, arr, pos_set_idx):
        n = arr.size(dim=0)
        c = arr.size(dim=1)
        neg = torch.zeros(n-1, c)
        
        # returns a tensor with the pos image removed
        if self.encoder_strategy == 1:
            for i in range(n):
                if i == pos_set_idx:
                    pass
                else:
                    neg[i, :] = arr[i, :]
        else:
            # TODO: implement neg pair sorting, need partition info
            pass
                
        return neg

    # implementation of eq 1 from the paper
    def individual_global_loss(self, latent1, latent2, neg_set):
        n = neg_set.size(0)
        # numerator term in eq 1
        num = torch.exp(self.cos_sim(latent1, latent2) / self.tau)
        # denominator term in eq 1
        denom = 0
        for i in range(n):
            denom += torch.exp(self.cos_sim(latent1, neg_set[i, :]))
        denom += num
        # return eq 1
        return -torch.log(torch.div(num, denom))

    # implementation of eq 2 from the paper
    def global_loss(self, aug1, aug2, unaug):
        loss = 0
        pos_set, aug_type = self.create_pos_set(aug1, aug2, unaug)
        n = pos_set.size(dim=0)
        # loop through all images in the pos set
        for i in range(n):
            if aug_type[i, 0] == 0:
                neg_set1 = self.create_neg_set(unaug, i)
            if aug_type[i, 0] == 1:
                neg_set1 = self.create_neg_set(aug1, i)
            if aug_type[i, 0] == 2:
                neg_set1 = self.create_neg_set(aug2, i)
            
            # pass pos_im1, pos_im2 and neg set to individual loss function<-- image order matters!
            l1 = self.individual_global_loss(pos_set[i, 1, :], pos_set[i, 2, :], neg_set1)
            
            if aug_type[i, 1] == 0:
                neg_set2 = self.create_neg_set(unaug, i)
            if aug_type[i, 1] == 1:
                neg_set2 = self.create_neg_set(aug1, i)
            if aug_type[i, 1] == 2:
                neg_set2 = self.create_neg_set(aug2, i)
            
            # pass pos_im2, pos_im1 and neg set to individual loss function <-- image order matters!
            l2 = self.individual_global_loss(pos_set[i, 2, :], pos_set[i, 1, :], neg_set2)
            
            loss += l1 + l2

        return loss / n

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


# testing with random inputs
if __name__ == "__main__":
    num_classes = 4
    loss = Loss()
    pred = torch.randn(8, num_classes, 192, 192)
    pred_softmax = F.softmax(pred, dim=1)

    target = torch.randint(low=0, high=4, size=(8, 1, 192, 192))
    print(f"unique output: {torch.unique(target)}")
    target_one_hot = loss.one_hot(target.long(), num_classes=num_classes)
    print(target_one_hot.shape)

    dice_loss = loss.compute(pred_softmax, target_one_hot, multiclass=True)
    print(f"dice loss: {dice_loss}")

    # in_channels = 1
    # num_filters = [1, 16, 32, 64, 128, 128]
    # fc_units = [3200, 1024]
    # g1_out_dim = 128
    # num_classes = 3

    # full_model = seg_models.SegUnetFullModel(in_channels, num_filters,fc_units, g1_out_dim, num_classes)
    # logits, output = full_model(torch.randn(8, 1, 192, 192))
    # print(logits.shape)
    # print(f"output shape: {output.shape}")