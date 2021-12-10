from operator import pos
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
    loss_type: 0 for dice loss (default), 1 for global loss
    encoder_strategy: 'gr' for Gr, 'gd-' for GD-, 'gd' for GD
'''
class Loss:
    def __init__(self, loss_type=0, encoder_strategy='gr', device='cpu'):
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

    def dice_loss(self, prediction, target):
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

        return 1.0 - torch.mean(dices)

    def contrastive_loss_simclr(self, proj_feat1, proj_feat2, temperature=0.5):
        """
        Adapted from: 
        https://github.com/colleenjg/neuromatch_ssl_tutorial/blob/130380eb77e46a993489d7c6c89d0c9ee8ce3ed3/modules/models.py#L358
        Returns contrastive loss, given sets of projected features, with positive pairs matched along the batch dimension.
        Required args:
            - proj_feat1 (2D torch Tensor): first set of projected features (batch_size x feat_size)
            - proj_feat2 (2D torch Tensor): second set of projected features (batch_size x feat_size)
        Optional args:
            - temperature (float): relaxation temperature. (default: 0.5)
        Returns:
            - loss (float): mean contrastive loss
        """
        # Normalize the individual representations
        batch_size = len(proj_feat1)
        z1 = F.normalize(proj_feat1, dim=1)
        z2 = F.normalize(proj_feat2, dim=1)

        # (vertical) stack one on top of the other
        representations = torch.cat([z1, z2], dim=0)  # shape: (2*batch-size) x g1_out_dimension

        # get the full similarity matrix 
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)  # shape: (2*batch-size) x (2*batch-size)

        # initialize arrays to set the indices of the positive and negative samples (shape: (2*batch-size) x (2*batch-size))
         # finds a positive sample (2*batch-size)//2 away from the original sample
        pos_sample_indicators = torch.roll(torch.eye(2*batch_size), batch_size, 1).to(proj_feat1.device)   
        neg_sample_indicators = (torch.ones(2*batch_size) - torch.eye(2*batch_size)).to(proj_feat1.device)

        # calculate the numerator by selecting the appropriate indices of the positive samples using the pos_sample_indicators matrix
        numerator = torch.exp(similarity_matrix/temperature)[pos_sample_indicators.bool()]      # shape: [2*batch_size]
        # calculate the denominator by summing over each pair except for the diagonal elements
        denominator = torch.sum((torch.exp(similarity_matrix/temperature)*neg_sample_indicators), dim=1)     # shape: [2*batch_size]

        # clamp to avoid division by zero
        if (denominator < 1e-8).any():
            denominator = torch.clamp(denominator, 1e-8)

        loss = torch.mean(-torch.log(numerator/denominator))
        return loss

    def loss_GDminus(self, proj_feat0, proj_feat1, proj_feat2, partition_size=4, temperature=0.5):
        """
        Required args:
            - proj_feat0 (2D torch Tensor): zero set of projected features (batch_size x feat_size) i.e. from the unaugmented image
            - proj_feat1 (2D torch Tensor): first set of projected features (batch_size x feat_size)
            - proj_feat2 (2D torch Tensor): second set of projected features (batch_size x feat_size)
            - partition_size (int): the number of partitions of each input volume (default = 4)
        Optional args:
            - temperature (float): relaxation temperature. (default: 0.5)
        Returns:
            - loss (float): mean contrastive loss
        """
        # Normalize the individual representations
        batch_size = len(proj_feat1)
        z0 = F.normalize(proj_feat0, dim=1)     # projected features from the unaugmented image
        z1 = F.normalize(proj_feat1, dim=1)
        z2 = F.normalize(proj_feat2, dim=1)
        N = 3*batch_size
        
        # (vertical) stack one on top of the other
        representations = torch.cat([z0, z1, z2], dim=0)  # shape: (3*batch-size) x g1_out_dimension

        # get the full similarity matrix 
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)  # shape: (3*batch-size) x (3*batch-size)

        # for GD-, the positive sample indices are similar to GR (simclr)
        # pos_sample_indicators = torch.roll(torch.eye(N), batch_size, 1) + torch.roll(torch.eye(N), 2*batch_size, 1)
        pos_sample_indicators_1 = torch.roll(torch.eye(N), batch_size, 1).to(proj_feat1.device) 
        pos_sample_indicators_2 = torch.roll(torch.eye(N), 2*batch_size, 1).to(proj_feat1.device)        
        # pos_sample_indicators = pos_sample_indicators.to(proj_feat1.device)
        # print(f"pos indicators shape: {pos_sample_indicators.shape}")
        
        # for the neg sample indices, we also need to consider the partition size here because we do not want to contrast against them
        # so for each row in the neg indicator matrix, we have the diagonal zeros (by default) and now we also have 0s spaced out 
        # according to the partition size as well (within each batch). This is replicated across the 3 batches
        neg_sample_indicators = torch.ones((N, N))
        for i in range(batch_size // partition_size):
            neg_sample_indicators = neg_sample_indicators 
            - torch.roll(torch.eye(N), i*partition_size, 1) 
            - torch.roll(torch.eye(N), i*partition_size + batch_size, 1) 
            - torch.roll(torch.eye(N), i*partition_size + 2*batch_size, 1)
        neg_sample_indicators = neg_sample_indicators.to(proj_feat1.device)

        # calculate the numerator by selecting the appropriate indices of the positive samples using the pos_sample_indicators matrix
        numerator_pos_1 = torch.exp(similarity_matrix/temperature)[pos_sample_indicators_1.bool()]      # shape: [3*batch-size]
        numerator_pos_2 = torch.exp(similarity_matrix/temperature)[pos_sample_indicators_2.bool()]      # shape: [3*batch-size]        
        # calculate the denominator by summing over each pair except for the diagonal elements
        denominator = torch.sum((torch.exp(similarity_matrix/temperature)*neg_sample_indicators), dim=1)

        # clamp to avoid division by zero
        if (denominator < 1e-8).any():
            denominator = torch.clamp(denominator, 1e-8)

        loss = torch.mean(-torch.log((numerator_pos_1 + numerator_pos_2)/denominator))
        return loss

    def loss_GD(self, proj_feat0, proj_feat1, proj_feat2, partition_size=4, temperature=0.5):
        """
        Required args:
            - proj_feat0 (2D torch Tensor): zero set of projected features (batch_size x feat_size) i.e. from the unaugmented image
            - proj_feat1 (2D torch Tensor): first set of projected features (batch_size x feat_size)
            - proj_feat2 (2D torch Tensor): second set of projected features (batch_size x feat_size)
            - partition_size (int): the number of partitions of each input volume (default = 4)
        Optional args:
            - temperature (float): relaxation temperature. (default: 0.5)
        Returns:
            - loss (float): mean contrastive loss
        """
        # Normalize the individual representations  (TODO: there should be 4 latents)
        batch_size = len(proj_feat1)
        z0 = F.normalize(proj_feat0, dim=1)     # projected features from the unaugmented image
        z1 = F.normalize(proj_feat1, dim=1)
        z2 = F.normalize(proj_feat2, dim=1)

        # (vertical) stack one on top of the other  (TODO: change to 4)
        representations = torch.cat([z0, z1, z2], dim=0)  # shape: (4*batch-size) x g1_out_dimension

        # get the full similarity matrix 
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)  # shape: (4*batch-size) x (4*batch-size)

        # initialize arrays to set the indices of the positive and negative samples (shape: (4*batch-size) x (4*batch-size))
        # here, the positive sample strategy is similar to neg samples in GD-, except where there are 0s in GD- (denoting samples to not 
        # constrast against), they become 1s in GD for positives
        N = 3*batch_size
        pos_sample_indicators_1 = torch.zeros(N)
        pos_sample_indicators_2 = torch.zeros(N)
        pos_sample_indicators_3 = torch.zeros(N)
        for i in range(batch_size // partition_size):
            pos_sample_indicators_1 = (pos_sample_indicators_1 + torch.roll(torch.eye(N), i*partition_size, 1)).to(proj_feat1.device) 
            pos_sample_indicators_2 = (pos_sample_indicators_2 + torch.roll(torch.eye(N), i*partition_size+batch_size, 1)).to(proj_feat1.device) 
            pos_sample_indicators_3 = (pos_sample_indicators_3 + torch.roll(torch.eye(N), i*partition_size+2*batch_size, 1)).to(proj_feat1.device) 

        # for i in range(batch_size // partition_size):
        #     pos_sample_indicators = pos_sample_indicators 
        #     + torch.roll(torch.eye(N), i*partition_size, 1) 
        #     + torch.roll(torch.eye(N), i*partition_size + batch_size, 1)
        #     + torch.roll(torch.eye(N), i*partition_size + 2*batch_size, 1)
        # pos_sample_indicators = pos_sample_indicators.to(proj_feat1.device)
        
        # the negative sample indices remain unchanged from GD-'s negative sample indices
        neg_sample_indicators = torch.ones(N)
        for i in range(batch_size // partition_size):
            neg_sample_indicators = neg_sample_indicators 
            - torch.roll(torch.eye(N), i*partition_size, 1) 
            - torch.roll(torch.eye(N), i*partition_size + batch_size, 1)
            - torch.roll(torch.eye(N), i*partition_size + 2*batch_size, 1)
        neg_sample_indicators = neg_sample_indicators.to(proj_feat1.device)
        # # shorter version
        # neg_sample_indicators = (~pos_sample_indicators.bool()).float() - torch.eye(N)

        # calculate the numerator by selecting the appropriate indices of the positive samples using the pos_sample_indicators matrix
        numerator_pos_1 = torch.exp(similarity_matrix/temperature)[pos_sample_indicators_1.bool()]
        numerator_pos_2 = torch.exp(similarity_matrix/temperature)[pos_sample_indicators_2.bool()]
        numerator_pos_3 = torch.exp(similarity_matrix/temperature)[pos_sample_indicators_3.bool()]                
        # calculate the denominator by summing over each pair except for the diagonal elements
        denominator = torch.sum((torch.exp(similarity_matrix/temperature)*neg_sample_indicators), dim=1)

        # clamp to avoid division by zero
        if (denominator < 1e-8).any():
            denominator = torch.clamp(denominator, 1e-8)

        # loss = torch.mean(-torch.log((numerator_pos_1 + numerator_pos_2 + numerator_pos_3)/denominator))
        loss1 = torch.mean(-torch.log(numerator_pos_1/denominator))
        loss2 = torch.mean(-torch.log(numerator_pos_2/denominator))
        loss3 = torch.mean(-torch.log(numerator_pos_3/denominator))
        loss = (loss1+loss2+loss3)/3.0
        return loss

    def cos_sim(self, vect1, vect2):
        vect1_norm = F.normalize(vect1, dim=-1, p=2)
        vect2_norm = torch.transpose(F.normalize(vect2, dim=-1, p=2), -1, -2)
        return torch.matmul(vect1_norm, vect2_norm)

    def create_pos_set(self, latent_mini_batch):
        # based on being passed output of g1(e())
        # image 3*n-3 is unaugmented, 3*n-2 is augmentation 1 and 3*n is augmentation 3*n-1
        # array augmentation_type contains on entry per sample specifying the augmentation type: 0=none, 1=type1, 2 = type2
        n, c = latent_mini_batch.shape

        if self.loss_type == 1:
            if self.encoder_strategy == 1:
                pos = torch.zeros(int(n/3), 2, c)
                augmentation_type = torch.zeros(int(n/3), 2)
            if self.encoder_strategy == 2:
                pos = torch.zeros(n, 2, c)
                augmentation_type = torch.zeros(n, 2)
            if self.encoder_strategy == 3:
                pos = torch.zeros((int(n/3)*4), 2, c)
                augmentation_type = torch.zeros(int((n/3)*4), 2)

            for i in range(1, int((n+3)/3)):
                if self.encoder_strategy == 1:
                    pairs = torch.zeros(1, 2, 128)
                    pairs[1, 1, :] = latent_mini_batch[(n * 3) - 2, :]
                    pairs[1, 2, :] = latent_mini_batch[(n * 3) - 1, :]
                    augmentation_type[n, :] = torch.tensor([1., 2.])
                    pos[n, :, :] = pairs

                if self.encoder_strategy == 2:
                    pairs = torch.zeros(3, 2, 128)
                    pairs[1, 1, :] = latent_mini_batch[(n * 3) - 2, :]
                    pairs[1, 2, :] = latent_mini_batch[(n * 3) - 1, :]
                    augmentation_type[n, :] = torch.tensor([1., 2.])
                    pairs[2, 1, :] = latent_mini_batch[(n * 3) - 3, :]
                    pairs[2, 2, :] = latent_mini_batch[(n * 3) - 2, :]
                    augmentation_type[n, :] = torch.tensor([0., 1.])
                    pairs[3, 1, :] = latent_mini_batch[(n * 3) - 3, :]
                    pairs[3, 2, :] = latent_mini_batch[(n * 3) - 1, :]
                    augmentation_type[n, :] = torch.tensor([0., 2.])

                    pos[n, :, :] = pairs
                if self.encoder_strategy == 3:
                    # TODO: figure out how to get 2nd image for pos pairs (need partition info)
                    pairs = torch.zeros(4, 2, 128)

        if self.loss_type == 2:
            pass

        return pos, augmentation_type

    def create_neg_set(self, latent_mini_batch, pos_set_idx, augmentation_type):
        n, c = latent_mini_batch.shape
        neg = torch.zeros(int((n / 3) - 1), c)

        for i in range(1, int((n + 3) / 3)):
            if i == pos_set_idx:
                pass
            else:
                if augmentation_type == 0:
                    neg[i, :] = latent_mini_batch[(n * 3) - 3, :]
                if augmentation_type == 1:
                    neg[i, :] = latent_mini_batch[(n * 3) - 2, :]
                if augmentation_type == 2:
                    neg[i, :] = latent_mini_batch[(n * 3) - 1, :]
        return neg

    def individual_global_loss(self, latent1, latent2, neg_set):
        n, c = neg_set.shape
        num = torch.exp(self.cos_sim(latent1, latent2) / self.tau)
        denom = 0
        for i in range(n):
            denom += torch.exp(self.cos_sim(latent1, neg_set[i, :]))
        denom += num
        return -torch.log(torch.div(num, denom))

    def global_loss(self, prediction):
        n, _, _ = prediction.shape
        loss = 0
        pos_set, augmentations = self.create_pos_set(prediction)
        m, _, _ = pos_set.shape
        for i in range(m):
            neg_set = self.create_neg_set(prediction, i, augmentations[i, 0])
            l1 = self.individual_global_loss(pos_set[i, 1, :], pos_set[i, 2, :], neg_set)
            neg_set = self.create_neg_set(prediction, i, augmentations[i, 1])
            l2 = self.individual_global_loss(pos_set[i, 2, :], pos_set[i, 1, :], neg_set)
            loss += l1+l2

        return loss / m

    def compute(self, proj_feat0, proj_feat1, proj_feat2, partition_size,
                prediction, target=None, multiclass=False):
        if self.loss_type == 0:
            prediction = prediction.to(self.device)
            target = target.to(self.device)
            return self.dice_loss_v2(prediction, target, multiclass)    # the new, "working" dice loss
            # return self.dice_loss(prediction, target, multiclass)     # original, incorrect one
        elif self.loss_type == 1:   # means global loss
            if self.encoder_strategy == 'gr':
                return self.contrastive_loss_simclr(proj_feat1, proj_feat2)
            elif self.encoder_strategy == 'gd-':
                return self.loss_GDminus(proj_feat0, proj_feat1, proj_feat2, partition_size)
            elif self.encoder_strategy == 'gd':
                return self.loss_GD(proj_feat0, proj_feat1, proj_feat2, partition_size)

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