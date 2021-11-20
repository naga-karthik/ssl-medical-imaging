import torch
import torch.nn.functional as f

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
