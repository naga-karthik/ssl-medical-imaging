import torch
import torch.nn.functional as f
import seg_models

'''
Inputs:
    loss_type: 0 for dice loss (default), 1 for global loss, 2 for local loss
    encoder_strategy: 0 for random (default), 1 for Gr, 2 dor Gd-, 3 for Gd
    decoder_strategy: 0 for random (default), 1 for Lr, 2 for Ld
'''
class Loss:
    def __init__(self, loss_type=0, encoder_strategy=0, decoder_strategy=0):
        self.decoder_strategy = decoder_strategy
        self.encoder_strategy = encoder_strategy
        self.loss_type = loss_type
        self.tau = 0.1
        self.smooth = 0.001     # smoothing factor for the dice loss

    def one_hot(self, arr, num_classes):
        return torch.eye(num_classes)[arr]

    def dice_loss(self, prediction, target, test=False):
        # should output number of unique classes
        classes = torch.unique(target)
        c = classes.shape[0]
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
        vect1_norm = f.normalize(vect1, dim=-1, p=2)
        vect2_norm = torch.transpose(f.normalize(vect2, dim=-1, p=2), -1, -2)
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


    def local_loss(self, prediction):
        # TODO: local loss implementation (L_l)
        pass

    def compute(self, prediction, target=None):
        if self.loss_type == 0:
            return self.dice_loss(prediction, target)
        if self.loss_type == 1:
            return self.global_loss(prediction)
        elif self.loss_type == 2:
            return self.local_loss(prediction)

# testing with random inputs
if __name__ == "__main__":
    in_channels = 1
    num_filters = [1, 16, 32, 64, 128, 128]
    fc_units = [3200, 1024]
    g1_out_dim = 128
    num_classes = 3

    full_model = seg_models.SegUnetFullModel(in_channels, num_filters,fc_units, g1_out_dim, num_classes)
    logits, output = full_model(torch.randn(8, 1, 192, 192))
    print(logits.shape)
    print(f"output shape: {output.shape}")

    test_labels = torch.randint(low=0, high=3, size=(8, 1, 192, 192))
    loss = Loss()
    # test one hot vectorization
    # print(loss.one_hot(test_labels.view(-1), 3))

    loss_result = loss.compute(output, test_labels)
    # loss_result = loss.compute(logits, test_labels)        # for dice_loss_v2
    print(loss_result)