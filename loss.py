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

    def dice_loss(self, prediction, target):
        print(f'prediction: {prediction.shape}, target: {target.shape}')
        pflat = prediction.view(-1)
        tflat = target.view(-1)
        intersection = (pflat * tflat).sum()
        score = (2.0 * intersection + 1.0) / (pflat.sum() + tflat.sum() + 1.0)

        return 1.0 - torch.mean(score)

    def cos_sim(self, vect1, vect2):
        vect1_norm = f.normalize(vect1, dim=-1, p=2)
        vect2_norm = torch.transpose(f.normalize(vect2, dim=-1, p=2), -1, -2)
        return torch.matmul(vect1_norm, vect2_norm)

    def create_pos_set(self, mini_batch):
        # based on nx1x192x192 images per batch: we have n images (n=vol*partition), r image width, c image height
        # assuming with augmentations minibatch dimensions will be 3nx1x192x192
        # image 3*n-2 is unaugmented, 3*n-1 is augmentation 1 and 3*n is augmentation 3*n
        n, l, r, c = mini_batch.shape

        if self.loss_type == 1:
            # dimensions: num_pos_combosxindividual_laternt_in_each_comboxlatent_dim
            if self.encoder_strategy == 1:
                pos = torch.zeros(n/3, 2, 128)
            if self.encoder_strategy == 2:
                pos = torch.zeros(n, 2, 128)
            if self.encoder_strategy == 3:
                pos = torch.zeros((n/3)*4, 2, 128)

            model = seg_models.SegUnetEncoder_and_ProjectorG1()
            latents = model(mini_batch)  # should be nx128 returned
            for i in range(n/3):
                if self.encoder_strategy == 1:
                    pairs = torch.zeros(1, 2, 128)
                    pairs[1, 1, :] = latents[(n * 3) + 1, :]
                    pairs[1, 2, :] = latents[(n * 3) + 2, :]
                    pos[n, :, :] = pairs

                if self.encoder_strategy == 2:
                    pairs = torch.zeros(3, 2, 128)
                    pairs[1, 1, :] = latents[(n * 3) + 1, :]
                    pairs[1, 2, :] = latents[(n * 3) + 2, :]
                    pairs[2, 1, :] = latents[(n * 3) + 0, :]
                    pairs[2, 2, :] = latents[(n * 3) + 1, :]
                    pairs[3, 1, :] = latents[(n * 3) + 0, :]
                    pairs[3, 2, :] = latents[(n * 3) + 2, :]
                    pos[n, :, :] = pairs
                if self.encoder_strategy == 3:
                    # TODO: figure out how to get 2nd image for pos pairs
                    pairs = torch.zeros(4, 2, 128)

        if self.loss_type == 2:
            pass

        return pos

    def individual_global_loss(self, latent1, latent2, neg_set, pair_order):
        n, l = neg_set.shape
        num = torch.exp(self.cos_sim(latent1, latent2) / self.tau)
        denom = 0
        for i in range(n):
            denom += torch.exp(self.cos_sim(latent1, neg_set[i, pair_order, :]))
        denom += num
        return -torch.log(torch.div(num, denom))

    def global_loss(self, pos_set):
        n, _, _ = pos_set.shape
        loss = 0
        if self.encoder_strategy == 1:
            neg_set = torch.zeros((n-1, 2, 128))
        if self.encoder_strategy == 2:
            # TODO: need partition membership info to implement
            pass
        if self.encoder_strategy == 3:
            # TODO: need partition membership info to implement
            pass

        for i in range(n):
            neg_set_idx = 0
            for j in range(n):
                if j != i:
                    neg_set[neg_set_idx, :, :] = pos_set[j, :, :]

            l1 = self.individual_global_loss(pos_set[i, 1, :], pos_set[i, 2, :], neg_set, pair_order=0)
            l2 = self.individual_global_loss(pos_set[i, 2, :], pos_set[i, 1, :], neg_set, pair_order=1)
            loss += l1+l2

        return loss / n


    def local_loss(self, prediction):
        # TODO: local loss implementation (L_l)
        pass

    def compute(self, prediction, target):
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
    num_classes = 1
    full_model = seg_models.SegUnetFullModel(in_channels, num_filters, fc_units, g1_out_dim, num_classes)
    _, output = full_model(mini_batch=torch.randn(8, 1, 192, 192))


    '''ground_truth_masks = torch.randn(8, 1, 192, 192)
    dice_loss = Loss(mini_batch=torch.randn(8, 1, 192, 192), loss_type=0, encoder_strategy=0, decoder_strategy=0)
    loss = dice_loss.compute(output, ground_truth_masks)
    print(f'computed loss: {loss}')'''