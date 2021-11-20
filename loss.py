import torch
import torch.nn.functional as f

class Loss:
    def __init__(self, loss_type):
        self.loss_type = loss_type

    def cos_sim(self, vect1, vect2):
        vect1_norm = f.normalize(vect1, dim=-1, p=2)
        vect2_norm = torch.transpose(f.normalize(vect2, dim=-1, p=2), -1, -2)
        return torch.matmul(vect1_norm, vect2_norm)
