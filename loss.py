import torch
import torch.nn.functional as f
import seg_models

'''loss_type is 0 for global, 1 for Gd- and 2 for Gd'''
class Loss:
    # TODO: figure out how to access e(.), g1(.), g2(.) and dl(.) so we can implement loss fn's.  Should it all be in one class??
    def __init__(self, loss_type, mini_batch):
        self.loss_type = loss_type
        self.tau = 0.1
        self.mini_batch = mini_batch

    def cos_sim(self, vect1, vect2):
        vect1_norm = f.normalize(vect1, dim=-1, p=2)
        vect2_norm = torch.transpose(f.normalize(vect2, dim=-1, p=2), -1, -2)
        return torch.matmul(vect1_norm, vect2_norm)

    def create_augmented_mini_batch_and_latents(self):
        # based on 4x1x192x192 image per volume: we have n volumes, s partitions, r image width, c image height
        n, s, l, r, c = self.mini_batch.shape
        augmented_images = torch.zeros(2*n, s, l, r, c)
        latents = torch.zeros((2 * n, s, 128))

        for vol in range(n):
            # TODO: select augmentations and augment.  output: augment1, augment2, the two different augmentations of the same source image
            augment1 = torch.randn(s, l, r, c) # placeholder
            augment2 = torch.randn(s, l, r, c)  # placeholder
            model = seg_models.SegUnetEncoder_and_ProjectorG1(in_channels=1, base_n_filter=4, g1_out_dim=128)
            _, latent1, _, _ = model(augment1)
            _, latent2, _, _ = model(augment2)
            augmented_images[vol * 2, :, :, :, :] = augment1
            augmented_images[(vol * 2) + 1, :, :, :, :] = augment2
            latents[vol*2, :, :] = latent1
            latents[(vol*2)+1, :, :] = latent2

        return augmented_images, latents

    def create_pos_neg(self):
        pos = torch.empty_like(self.mini_batch)
        neg = torch.empty_like(self.mini_batch)


        # TODO: create pos and neg samples, need to access augmentations or will they be pre computed in the passed minibatch?
        if self.loss_type == 1:
            # TODO: strategy to create pos pairs for Gd-
            pass
        elif self.loss_type == 2:
            # TODO: strategy to create pos pairs for Gd
            pass
        else:
            # TODO: strategy to create pos pairs for global loss
            pass
        return pos, neg

    def compute(self):
        if self.loss_type == 1:
            # TODO: local loss implementation Gd- (L_l)
            pass
        elif self.loss_type == 2:
            # TODO: local loss implementation Gd (L_l)
            pass
        else:
            # TODO: global loss implementation (L_g)
            pass

# testing with random inputs
if __name__ == "__main__":
    # assuming 4 volumes, 4 partitions, 1 label, 192 image width, 192 image height
    global_loss = Loss(loss_type=0, mini_batch=torch.randn(4, 4, 1, 192, 192))
    augmented, latents = global_loss.create_augmented_mini_batch_and_latents()
    print(f'interested in projector output should be 8x4x128.  result: {latents.shape}')