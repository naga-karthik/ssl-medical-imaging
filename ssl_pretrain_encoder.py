# utility packages
from functools import cmp_to_key
import os
import time
import argparse
from torch._C import device

import numpy as np
from torch.nn.modules.module import T
import matplotlib.pyplot as plt
from torch.utils.data.sampler import WeightedRandomSampler
timestamp = time.time()

# machine learning packages
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader, dataloader
import torchvision.transforms as transforms

# dataloaders and segmentation models
from seg_models_v2 import UNetEncoder, ProjectorHead
from Dataset.init_data import acdc, md_prostate
from Dataset.dataset import DatasetGR, DatasetGDMinus
from Dataset.experiments_paper import data_init_acdc, data_init_prostate_md
from loss import Loss, multiclass_dice_coeff

img_path = "/home/GRAMES.POLYMTL.CA/u114716/ssl_project/datasets/ACDC"
seg_path = "/home/GRAMES.POLYMTL.CA/u114716/ssl_project/datasets/ACDC"

parser = argparse.ArgumentParser(description="Random-Random Strategy Run 3")

# all the arguments for the dataset, model, and training hyperparameters
parser.add_argument('--exp_name', default='GR_Pretrain', type=str, help='Name of the experiment/run')
# dataset
parser.add_argument('-data', '--dataset', default=acdc, help='Specifyg acdc or md_prostate without quotes')
parser.add_argument('-nti', '--num_train_imgs', default='tr52', type=str, help='Number of training images, options tr1, tr8 or tr52')
parser.add_argument('-cti', '--comb_train_imgs', default='c1', type=str, help='Combintation of Train imgs., options c1, c2, cr3, cr4, cr5')
parser.add_argument('--img_path', default=img_path, type=str, help='Absolute path of the training data')
parser.add_argument('--seg_path', default=seg_path, type=str, help='Same as path of training data')
# model
parser.add_argument('-in_ch', '--in_channels', default=1, type=int, help='Number of input channels')
# parser.add_argument('-num_flt', '--num_filters_list', nargs='+', default=[16, 32, 64, 128, 256, 512], help='List containing no. of filters for Conv Layers')
# parser.add_argument('-num_flt', '--num_filters_list', nargs='+', default=[1, 16, 32, 64, 128, 128], help='List containing no. of filters for Conv Layers')
parser.add_argument('-num_flt', '--init_num_filters', type=int, default=32, help='Initial no. of filters for Conv Layers')
parser.add_argument('-num_fc', '--fc_units_list', nargs='+', default=[3200, 1024], help='List containing no. of units in FC layers')
parser.add_argument('-g1_dim', '--g1_out_dim', default=128, type=int, help='Output dimension for the projector head')
parser.add_argument('-nc', '--num_classes', default=4, type=int, help='Number of classes to segment')
# optimization
parser.add_argument('-p', '--precision', default=32, type=int, help='Precision for training')
parser.add_argument('-ep', '--epochs', default=250, type=int, help='Number of epochs to train')
parser.add_argument('-bs', '--batch_size', default=64, type=int, help='Batch size')
parser.add_argument('-nw', '--num_workers', default=4, type=int, help='Number of worker processes')
parser.add_argument('-gpus', '--num_gpus', default=1, type=int, help="Number of GPUs to use")
parser.add_argument('-lr', '--learning_rate', default=1e-3, type=float, help="Learning rate to use")
parser.add_argument('-wd', '--weight_decay', default=0.0, type=float, help='Default weight decay')

cfg = parser.parse_args()

class SegModel(pl.LightningModule):
    def __init__(self, cfg):
        super(SegModel, self).__init__()
        self.cfg = cfg
        self.save_hyperparameters()

        # self.net = UNet(n_channels=self.cfg.in_channels, init_filters=self.cfg.init_num_filters, n_classes=self.cfg.num_classes)
        self.encoder = UNetEncoder(n_channels=self.cfg.in_channels, init_filters=self.cfg.init_num_filters)
        self.projector = ProjectorHead(encoder_init_filters=self.cfg.init_num_filters, out_dim=self.cfg.g1_out_dim)

        self.train_ids_acdc = data_init_acdc.train_data(self.cfg.num_train_imgs, self.cfg.comb_train_imgs)
        self.val_ids_acdc = data_init_acdc.val_data(self.cfg.num_train_imgs, self.cfg.comb_train_imgs)
        # self.test_ids_acdc = data_init_acdc.test_data()

        self.train_dataset = DatasetGR(self.cfg.dataset, self.train_ids_acdc, self.cfg.img_path, preprocessed_data=True, seg_path=None)
        self.valid_dataset = DatasetGR(self.cfg.dataset, self.val_ids_acdc, self.cfg.img_path, preprocessed_data=True, seg_path=None)

        self.loss = Loss(loss_type=1, encoder_strategy=1, device=self.device)
        
    def forward(self, x):
        return self.net(x)

    def compute_loss(self, batch):
        img_aug1, img_aug2 = batch
        img_aug1, img_aug2 = img_aug1.float(), img_aug2.float()

        # get the latent representations by passing each augmented image through the encoder
        latent_reps_aug1 = self.encoder(img_aug1)
        latent_reps_aug2 = self.encoder(img_aug2)
        # get the final z's for the contrastive loss by passing through the projector head
        z_aug1 = self.projector(latent_reps_aug1)
        z_aug2 = self.projector(latent_reps_aug2)

        contrastive_loss = self.loss.compute(proj_feat1=z_aug1, proj_feat2=z_aug2, prediction=None)

        return contrastive_loss
    
    def training_step(self, batch, batch_nb):
        loss = self.compute_loss(batch)
        self.log('train_contrastive_loss', loss, on_step=False, on_epoch=True)

        # if batch_nb == 0: # once per epoch
        #     fig = visualize(preds, imgs, gts)
        #     wandb.log({"Training Output Visualizations": fig})
        return loss

    def validation_step(self, batch, batch_nb):
        loss = self.compute_loss(batch)
        self.log('valid_contrastive_loss', loss, on_step=False, on_epoch=True)

        # if batch_nb == 0: # once per epoch
        #     fig = visualize(preds, imgs, gts)
        #     wandb.log({"Validation Output Visualizations": fig})
    
    def test_step(self, batch, batch_nb):
        loss, preds, imgs, gts = self.compute_loss(batch)
        self.log('test_loss', loss, on_step=False, on_epoch=True)

        # dice score (found online)
        test_gt_one_hot = self.loss.one_hot(gts.long(), num_classes=self.cfg.num_classes)
        # computing the dice score, ignoring the background ie class 0
        test_dice_score = multiclass_dice_coeff(
            (preds[:, 1:, :, :]).to(self.device), 
            (test_gt_one_hot[:, 1:, :, :]).to(self.device), 
            reduce_batch_first=False
        )
        print(f"\nDICE SCORE ON THE TEST SET: {test_dice_score}")

        # qualitative results on wandb
        fig = visualize(preds, imgs, gts)
        wandb.log({"Test Output Visualizations": fig})
        test_img, test_gt = batch
        test_img = test_img.float()
        
        return {'test_dice_score' : test_dice_score}

    def configure_optimizers(self):
        # opt_params = { 'lr': 1e-4, 'weight_decay': 0, }
        # scheduler_params={ 'T_0': 40, 'eta_min': 1e-5 }
        # optimizer = eval(self.cfg.opt)(self.parameters(), **self.cfg.opt_params)
        # scheduler = eval(self.cfg.scheduler)(optimizer, **self.cfg.scheduler_params)

        optimizer = optim.Adam(params=self.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=40, eta_min=1e-5)
        
        return [optimizer], [scheduler]
        # return [optimizer]
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.cfg.batch_size,
                             shuffle = True, drop_last=True, num_workers=self.cfg.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size = self.cfg.batch_size,
                             shuffle = False, drop_last=False, num_workers=self.cfg.num_workers)
    
    # def test_dataloader(self):
    #     return DataLoader(self.test_dataset, batch_size = self.cfg.batch_size,
    #                          shuffle = False, drop_last=False, num_workers=self.cfg.num_workers)

def visualize(preds, imgs, gts, num_imgs=10):
    main_colors = torch.tensor([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]).view(8, 3).float()
    # getting ready for post processing
    imgs, gts, preds = imgs.detach().cpu(), gts.detach().cpu(), preds.detach().cpu()
    imgs = imgs.squeeze(dim=1).numpy()
    gts = gts.squeeze(dim=1)

    num_classes = preds.shape[1]
    colors = main_colors[:num_classes]
    # coloring the predictions
    preds[preds < torch.max(preds, dim=1, keepdims=True)[0]] = 0
    preds_colored = torch.tensordot(preds, colors, dims=[[1], [0]]).numpy()
    # coloring the ground truth masks
    gts_onehot = F.one_hot(gts, num_classes=num_classes).permute(0, 3, 1, 2)
    gts_colored = torch.tensordot(gts_onehot.float(), colors, dims=[[1], [0]]).numpy()

    fig, axs = plt.subplots(3, num_imgs, figsize=(9, 3))
    fig.suptitle('Original --> Ground Truth --> Prediction')
    for i in range(num_imgs):
        img_num = np.random.randint(0, len(imgs))
        axs[0, i].imshow(imgs[img_num], cmap='gray'); axs[0, i].axis('off') 
        axs[1, i].imshow(gts_colored[img_num]); axs[1, i].axis('off')    
        axs[2, i].imshow(preds_colored[img_num]); axs[2, i].axis('off')
    fig.show()
    return fig

def main(cfg):
    # experiment tracker (you need to sign in with your account)
    wandb_logger = pl.loggers.WandbLogger(
                            name='%s <- %d'%(cfg.exp_name, timestamp), 
                            group= '%s'%(cfg.exp_name), 
                            log_model=True, # save best model using checkpoint callback
                            project='self-supervised-pretrain',
                            entity='ssl-medical-imaging',
                            config=cfg,
    )

    # to save the best model on validation
    checkpoint = pl.callbacks.ModelCheckpoint(
        filename="best_model"+str(timestamp),
        monitor="valid_contrastive_loss",
        save_top_k=1,
        mode="max",
        save_last=False,
        save_weights_only=True,
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        devices=cfg.num_gpus, accelerator="gpu", strategy="ddp",
        logger=wandb_logger,
        callbacks=[checkpoint, lr_monitor],
        max_epochs=cfg.epochs,
        precision=cfg.precision,
    )

    model = SegModel(cfg)
    # log gradients, parameter histogram and model topology
    wandb_logger.watch(model, log='all')
    
    trainer.fit(model)
    print("------- Training Done! -------")

    # print("------- Saving the Best Model! -------")
    # torch.save(model.state_dict(), save_path)

    # print("------- Loading the Best Model! ------")     # the standard PyTorch Way
    # # load the best checkpoint after training
    # loaded_model = model.load_state_dict(torch.load(save_path), strict=False) # .load_from_checkpoint(trainer.checkpoint_callback.best_model_path, strict=False)
    # pretrained_encoder = loaded_model.encoder
    # pretrained_encoder.eval()
    # print(f"Pretrained Encoder model: \n{pretrained_encoder} ")

    save_path = "./best_encoder_model.pt"    # current folder    
    print("------- Loading the Best Model! ------")     # the PyTorch Lightning way
    # load the best checkpoint after training
    loaded_model = model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, strict=False)
    pretrained_encoder = loaded_model.encoder
    # pretrained_encoder.eval()
    # print(f"Pretrained Encoder model: \n{pretrained_encoder} ")    
    torch.save(pretrained_encoder.state_dict(), save_path)

    # print("------- Testing Begins! -------")
    # trainer.test(model)

if __name__ == '__main__':
    main(cfg)