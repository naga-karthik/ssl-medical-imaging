import torch
from torch._C import device
import torch.nn.functional as F
from torch import Tensor

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
    encoder_strategy: 'gr' for Gr, 'gd-' for GD-, 'gd' for GD, 'gd-alt' for the alternative GD/GDminus strategy
'''
class Loss:
    def __init__(self, loss_type=0, encoder_strategy='GR', device='cpu'):
        self.encoder_strategy = encoder_strategy
        self.loss_type = loss_type
        self.tau = 0.1          # temperature parameter for the contrastive loss
        self.smooth = 0.001     # smoothing factor for the dice loss
        self.device = device

    def dice_loss_v2(self, input: Tensor, target: Tensor, multiclass: bool = False):
        # Dice loss (objective to minimize) between 0 and 1
        assert input.size() == target.size()
        fn = multiclass_dice_coeff if multiclass else dice_coeff
        return 1 - fn(input, target, reduce_batch_first=True)
    
    def one_hot(self, arr, num_classes):
        # converting arr into a LongTensor so that it can be used as indices
        arr = arr.squeeze()
        one_hot_encoded = torch.eye(num_classes)[arr]   # shape: [batch_size, 192, 192, num_classes]
        return one_hot_encoded.permute(0, 3, 1, 2)      # shape: [batch_size, num_classes, 192, 192]

    def loss_GR(self, proj_feat1, proj_feat2):
        """
        Adapted from: 
        https://github.com/colleenjg/neuromatch_ssl_tutorial/blob/130380eb77e46a993489d7c6c89d0c9ee8ce3ed3/modules/models.py#L358
        Returns contrastive loss, given sets of projected features, with positive pairs matched along the batch dimension.
        Required args:
            - proj_feat1 (2D torch Tensor): first set of projected features (batch_size x feat_size)
            - proj_feat2 (2D torch Tensor): second set of projected features (batch_size x feat_size)
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
        numerator = torch.exp(similarity_matrix/self.tau)[pos_sample_indicators.bool()]      # shape: [2*batch_size]
        # calculate the denominator by summing over each pair except for the diagonal elements
        denominator = torch.sum((torch.exp(similarity_matrix/self.tau)*neg_sample_indicators), dim=1)     # shape: [2*batch_size]

        # clamp to avoid division by zero
        if (denominator < 1e-8).any():
            denominator = torch.clamp(denominator, 1e-8)

        loss = torch.mean(-torch.log(numerator/denominator))
        return loss

    def loss_GDminus(self, proj_feat0, proj_feat1, proj_feat2, partition_size=4):
        """
        Required args:
            - proj_feat0 (2D torch Tensor): zero set of projected features (batch_size x feat_size) i.e. from the unaugmented image
            - proj_feat1 (2D torch Tensor): first set of projected features (batch_size x feat_size)
            - proj_feat2 (2D torch Tensor): second set of projected features (batch_size x feat_size)
            - partition_size (int): the number of partitions of each input volume (default = 4)
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

        # # METHOD 1
        # # for GD-, the positive sample indices are similar to GR (simclr)
        # # pos_sample_indicators = torch.roll(torch.eye(N), batch_size, 1) + torch.roll(torch.eye(N), 2*batch_size, 1)
        # pos_sample_indicators_1 = torch.roll(torch.eye(N), batch_size, 1).to(proj_feat1.device) 
        # pos_sample_indicators_2 = torch.roll(torch.eye(N), 2*batch_size, 1).to(proj_feat1.device)        
        # # print(f"pos indicators shape: {pos_sample_indicators.shape}")
        
        # # for the neg sample indices, we also need to consider the partition size here because we do not want to contrast against them
        # # so for each row in the neg indicator matrix, we have the diagonal zeros (by default) and now we also have 0s spaced out 
        # # according to the partition size as well (within each batch). This is replicated across the 3 batches
        # neg_sample_indicators = torch.ones((N, N))
        # for i in range(batch_size // partition_size):
        #     neg_sample_indicators = neg_sample_indicators \
        #         - torch.roll(torch.eye(N), i*partition_size, 1) \
        #         - torch.roll(torch.eye(N), i*partition_size + batch_size, 1) \
        #         - torch.roll(torch.eye(N), i*partition_size + 2*batch_size, 1)
        # neg_sample_indicators = neg_sample_indicators.to(proj_feat1.device)

        # # calculate the numerator by selecting the appropriate indices of the positive samples using the pos_sample_indicators matrix
        # numerator_pos_1 = torch.exp(similarity_matrix/self.tau)[pos_sample_indicators_1.bool()]      # shape: [3*batch-size]
        # numerator_pos_2 = torch.exp(similarity_matrix/self.tau)[pos_sample_indicators_2.bool()]      # shape: [3*batch-size]        
        # # calculate the denominator by summing over each pair except for the diagonal elements
        # denominator = torch.sum((torch.exp(similarity_matrix/self.tau)*neg_sample_indicators), dim=1)

        # # clamp to avoid division by zero
        # if (denominator < 1e-8).any():
        #     denominator = torch.clamp(denominator, 1e-8)

        # loss = torch.mean(-torch.log((numerator_pos_1*numerator_pos_2)/denominator))

        # METHOD 2
        pos_sample_indicators_1 = torch.roll(torch.eye(N), batch_size, 1).to(proj_feat1.device) 
        pos_sample_indicators_2 = torch.roll(torch.eye(N), 2 * batch_size, 1).to(proj_feat1.device) 

        neg_sample_indicators = torch.ones(N, N)
        for i in range(batch_size // partition_size):
            neg_sample_indicators = neg_sample_indicators \
                                    - torch.roll(torch.eye(N), i * partition_size, 1) \
                                    - torch.roll(torch.eye(N), i * partition_size + batch_size, 1) \
                                    - torch.roll(torch.eye(N), i * partition_size + 2 * batch_size, 1)
        neg_sample_indicators = neg_sample_indicators.to(proj_feat1.device)

        # adding the corresponding positive sample indices to the denominator
        neg_sample_indicators_1 = neg_sample_indicators + pos_sample_indicators_1
        neg_sample_indicators_2 = neg_sample_indicators + pos_sample_indicators_2
        
        numerator_pos_1 = torch.exp(similarity_matrix)[pos_sample_indicators_1.bool()]  # shape: [3*batch-size]
        numerator_pos_2 = torch.exp(similarity_matrix)[pos_sample_indicators_2.bool()]  # shape: [3*batch-size]
        # calculate the denominator by summing over each pair except for the diagonal elements
        denominator_pos_1 = torch.sum((torch.exp(similarity_matrix) * neg_sample_indicators_1), dim=1)
        denominator_pos_2 = torch.sum((torch.exp(similarity_matrix) * neg_sample_indicators_2), dim=1)

        loss1 = torch.mean(-torch.log(numerator_pos_1 / denominator_pos_1))
        loss2 = torch.mean(-torch.log(numerator_pos_2 / denominator_pos_2))
        loss = 0.5*(loss1+loss2)

        return loss

    def GDminus_alt_helper(self, featA, featB, partition_size=4):
        """
        Alternative interpretation of the GDminus strategy of the paper. Instead of calculating a similarity matrix of 3x 
        the batch-size, 3 individual matrix (each of 2x batch-size) are calculated for each pair of the original and the 2
        augmented versions of the image. 
        Computed by borrowing positive sampling from SimCLR and negative sampling from GD- description in the paper.
        """
        # Normalize the individual representations
        batch_size = len(featA)
        z1 = F.normalize(featA, dim=1)
        z2 = F.normalize(featB, dim=1)
        N = 2*batch_size

        # (vertical) stack one on top of the other
        representations = torch.cat([z1, z2], dim=0)  # shape: (2*batch-size) x g1_out_dimension

        # get the full similarity matrix 
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)  # shape: (2*batch-size) x (2*batch-size)

        # initialize arrays to set the indices of the positive and negative samples (shape: (2*batch-size) x (2*batch-size))
         # finds a positive sample (2*batch-size)//2 away from the original sample
        pos_sample_indicators = torch.roll(torch.eye(N), batch_size, 1).to(featB.device)

        neg_sample_indicators = torch.ones((N, N))
        for i in range(batch_size // partition_size):
            neg_sample_indicators = neg_sample_indicators \
                - torch.roll(torch.eye(N), i*partition_size, 1) \
                - torch.roll(torch.eye(N), i*partition_size + batch_size, 1) \
                - torch.roll(torch.eye(N), i*partition_size + 2*batch_size, 1)
        neg_sample_indicators = neg_sample_indicators.to(featB.device)

        numerator = torch.exp(similarity_matrix/self.tau)[pos_sample_indicators.bool()]      # shape: [2*batch_size]
        denominator = torch.sum((torch.exp(similarity_matrix/self.tau)*neg_sample_indicators), dim=1)     # shape: [2*batch_size]

        # clamp to avoid division by zero
        if (denominator < 1e-8).any():
            denominator = torch.clamp(denominator, 1e-8)

        loss = torch.mean(-torch.log(numerator/denominator))
        return loss

    def loss_GDminus_alt(self, proj_feat0, proj_feat1, proj_feat2, partition_size=4):
        """
        Required args:
            - proj_feat0 (2D torch Tensor): zero set of projected features (batch_size x feat_size) i.e. from the unaugmented image
            - proj_feat1 (2D torch Tensor): first set of projected features (batch_size x feat_size)
            - proj_feat2 (2D torch Tensor): second set of projected features (batch_size x feat_size)
        Returns:
            - loss (float): mean contrastive loss
        """
        loss0 = self.GDminus_alt_helper(proj_feat0, proj_feat1, partition_size) 
        loss1 = self.GDminus_alt_helper(proj_feat0, proj_feat2, partition_size)
        loss2 = self.GDminus_alt_helper(proj_feat1, proj_feat2, partition_size)

        total_loss = (loss0 + loss1 + loss2)/3.0

        return total_loss

    def loss_GD(self, proj_feat0, proj_feat1, proj_feat2, partition_size=4):
        """
        Required args:
            - proj_feat0 (2D torch Tensor): zero set of projected features (batch_size x feat_size) i.e. from the unaugmented image
            - proj_feat1 (2D torch Tensor): first set of projected features (batch_size x feat_size)
            - proj_feat2 (2D torch Tensor): second set of projected features (batch_size x feat_size)
            - partition_size (int): the number of partitions of each input volume (default = 4)
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
        numerator_pos_1 = torch.exp(similarity_matrix/self.tau)[pos_sample_indicators_1.bool()]
        numerator_pos_2 = torch.exp(similarity_matrix/self.tau)[pos_sample_indicators_2.bool()]
        numerator_pos_3 = torch.exp(similarity_matrix/self.tau)[pos_sample_indicators_3.bool()]                
        # calculate the denominator by summing over each pair except for the diagonal elements
        denominator = torch.sum((torch.exp(similarity_matrix/self.tau)*neg_sample_indicators), dim=1)

        # clamp to avoid division by zero
        if (denominator < 1e-8).any():
            denominator = torch.clamp(denominator, 1e-8)

        # loss = torch.mean(-torch.log((numerator_pos_1 + numerator_pos_2 + numerator_pos_3)/denominator))
        loss = torch.mean(-torch.log((numerator_pos_1 * numerator_pos_2 * numerator_pos_3)/denominator))
        return loss

    def compute(self, proj_feat0, proj_feat1, proj_feat2, partition_size, prediction, target=None, multiclass=False):
        """
        Computes the loss function (dice or contrastive) depending on the pretraining encoder strategy or finetuning
        Required args just for pretraining:
        - Note: the remaining arguments must be set to None
            - proj_feat0 (2D torch Tensor): zero set of projected features (batch_size x feat_size) i.e. from the unaugmented image
            - proj_feat1 (2D torch Tensor): first set of projected features (batch_size x feat_size)
            - proj_feat2 (2D torch Tensor): second set of projected features (batch_size x feat_size)
            - partition_size (int): the number of partitions of each input volume (default = 4)
        Required args for finetuning/training from scratch:
        - Note: the remaining arguments must be set to None
            - prediction: the model prediction (softmax output)
            - target: one-hot encoded ground truth labels
            - multiclass (bool): True for multiclass classification, False for binary classification 
        Returns:
            - the corresponding loss function 
        """
        if self.loss_type == 0:
            prediction = prediction.to(self.device)
            target = target.to(self.device)
            return self.dice_loss_v2(prediction, target, multiclass)    # the new, "working" dice loss
        elif self.loss_type == 1: 
            if self.encoder_strategy == 'GR':
                return self.loss_GR(proj_feat1, proj_feat2)
            elif self.encoder_strategy == 'GD-':
                return self.loss_GDminus(proj_feat0, proj_feat1, proj_feat2, partition_size)
            elif self.encoder_strategy == 'GD':
                return self.loss_GD(proj_feat0, proj_feat1, proj_feat2, partition_size)
            elif self.encoder_strategy == 'GD-alt':
                return self.loss_GDminus_alt(proj_feat0, proj_feat1, proj_feat2, partition_size)

            
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