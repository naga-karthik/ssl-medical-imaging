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
        n, _ = prediction.shape
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


    def local_loss(self, prediction):
        # TODO: local loss implementation (L_l)
        pass

    def compute(self, prediction, target=None, multiclass=False):
        if self.loss_type == 0:
            prediction = prediction.to(self.device)
            target = target.to(self.device)
            return self.dice_loss_v2(prediction, target, multiclass)    # the new, "working" dice loss
            # return self.dice_loss(prediction, target, multiclass)     # original, incorrect one
        if self.loss_type == 1:
            return self.global_loss(prediction)
        elif self.loss_type == 2:
            return self.local_loss(prediction)


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