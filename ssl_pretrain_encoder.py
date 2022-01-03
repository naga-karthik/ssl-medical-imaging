# utility packages
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
from Dataset.dataset import DatasetGR, DatasetGD
from Dataset.experiments_paper import data_init_acdc, data_init_prostate_md
from loss import Loss, multiclass_dice_coeff

# define paths for the images and the segmentation labels
img_path = "/home/ssl_project/datasets/ACDC"
seg_path = "/home/ssl_project/datasets/ACDC"

parser = argparse.ArgumentParser(description="Random-Random Strategy Run 3")

# all the arguments for the dataset, model, and training hyperparameters
parser.add_argument('--exp_name', default='GR Strategy', type=str, help='Name of the experiment/run')
parser.add_argument('-st', '--strategy', default='GR', type=str, help='Strategy for pretraining; Options: GR, GD-, GD, GD-alt')
# dataset
parser.add_argument('-data', '--dataset', default='ACDC', help='Specifyg acdc or md_prostate without quotes')
parser.add_argument('--dataset_name', default='acdc', type=str, help='acdc or md_prostate dataset')
parser.add_argument('-nti', '--num_train_imgs', default='tr52', type=str, help='Number of training images, options tr1, tr2 or tr8')
parser.add_argument('-cti', '--comb_train_imgs', default='c1', type=str, help='Combintation of Train imgs., options c1, c2, c3, c4, c5')
parser.add_argument('--img_path', default=img_path, type=str, help='Absolute path of the training data')
parser.add_argument('--seg_path', default=seg_path, type=str, help='Same as path of training data')
# model
parser.add_argument('-in_ch', '--in_channels', default=1, type=int, help='Number of input channels')
parser.add_argument('-num_flt', '--init_num_filters', type=int, default=16, help='Initial no. of filters for Conv Layers')
parser.add_argument('-g1_dim', '--g1_out_dim', default=128, type=int, help='Output dimension for the projector head')
parser.add_argument('-nc', '--num_classes', default=4, type=int, help='Number of classes to segment')
parser.add_argument('-np', '--num_partitions', default=4, type=int, help='No. of partitions per volume')
# optimization
parser.add_argument('-p', '--precision', default=32, type=int, help='Precision for training')
parser.add_argument('-ep', '--epochs', default=10000, type=int, help='Number of epochs to train')
parser.add_argument('-bs', '--batch_size', default=12, type=int, help='Batch size')
parser.add_argument('-nw', '--num_workers', default=4, type=int, help='Number of worker processes')
parser.add_argument('-gpus', '--num_gpus', default=1, type=int, help="Number of GPUs to use")
parser.add_argument('-lr', '--learning_rate', default=5e-4, type=float, help="Learning rate to use")
parser.add_argument('-wd', '--weight_decay', default=1e-3, type=float, help='Default weight decay')
parser.add_argument('-pat', '--patience', default=20, type=int, help='number of validation steps (val_every_n_iters) to wait before early stoping')
parser.add_argument('--T_0', default=500, type=int, help='number of steps in each cosine cycle')
parser.add_argument('-epb', '--enable_progress_bar', default=False, type=bool, help='by default is disabled since it doesnt work in colab')
parser.add_argument('-chkp_pth', '--checkpoint_path', default='random-init', type=str, help='full path for the checkpoint in case would like to finetune')
parser.add_argument('--val_every_n_iters', default='100', type=int, help='num of iterations before validation')

cfg = parser.parse_args()

class SegModel(pl.LightningModule):
    def __init__(self, cfg):
        super(SegModel, self).__init__()
        self.cfg = cfg
        self.save_hyperparameters()

        if cfg.dataset == 'ACDC':
          data_init = data_init_acdc
          dataset_cfg = acdc 
        elif cfg.dataset  == 'MD_PROSTATE':
          data_init = data_init_prostate_md
          dataset_cfg = md_prostate
        else:
          print('The dataset is not found')
        self.num_class = dataset_cfg['num_class']

        self.encoder = UNetEncoder(n_channels=self.cfg.in_channels, init_filters=self.cfg.init_num_filters)
        self.projector = ProjectorHead(encoder_init_filters=self.cfg.init_num_filters, out_dim=self.cfg.g1_out_dim)

        self.train_ids = data_init.train_data(self.cfg.num_train_imgs, self.cfg.comb_train_imgs)
        self.val_ids = data_init.val_data(self.cfg.num_train_imgs, self.cfg.comb_train_imgs)
        # self.test_ids = data_init_acdc.test_data()

        if self.cfg.strategy == "GR":
            # Comment/Uncomment these for pretraining the encoder with the GR (SimCLR) strategy
            print("LOADING THE GR DATASET!")
            self.train_dataset = DatasetGR(dataset_cfg, self.train_ids, self.cfg.img_path, preprocessed_data=False, seg_path=None)
            self.valid_dataset = DatasetGR(dataset_cfg, self.val_ids, self.cfg.img_path, preprocessed_data=False, seg_path=None)
        else:
            # Comment/Uncomment these for pretraining the encoder with the GDminus, GD, and GD-alt strategies
            print("LOADING THE GD DATASET!")
            self.train_dataset = DatasetGD(dataset_cfg, self.train_ids, self.cfg.num_partitions, self.cfg.img_path, preprocessed_data=False, seg_path=None)
            self.valid_dataset = DatasetGD(dataset_cfg, self.val_ids, self.cfg.num_partitions, self.cfg.img_path, preprocessed_data=False, seg_path=None)

        # Choosing the loss function
        self.loss = Loss(loss_type=1, encoder_strategy=self.cfg.strategy, device=self.device)    # also make a change on line 278 for saving with the correct file name    
        self.loss_visualization_step = 0.1
        self.best_valid_loss = 1
        self.best_train_loss = 1

        self.train_losses, self.valid_losses = [], []
        self.init_timestamp = time.time()
        self.num_iters_per_epoch = np.int(np.ceil(len(self.train_dataset) / self.cfg.batch_size))
        
        print("--------------------------------------------------------------")
        print(f"PRETRAINING THE ENCODER WITH THE {self.cfg.strategy} STRATEGY!")
        print("--------------------------------------------------------------")
        
    def forward(self, x):
        return self.net(x)

    def compute_loss(self, batch, strategy):
        """
        Loss function for pretraining the encoder with various contrastive strategies 
        """
        if strategy == 'GR':
            img_aug1, img_aug2 = batch
            img_aug1, img_aug2 = img_aug1.float(), img_aug2.float()

            # get the latent representations by passing each augmented image through the encoder
            # latent_reps_aug1 = self.encoder(img_aug1)   # this is storing them as tuple; encoder has 2 outputs
            # latent_reps_aug2 = self.encoder(img_aug2)
            latent_reps_aug1, _ = self.encoder(img_aug1)
            latent_reps_aug2, _ = self.encoder(img_aug2)
            # get the final z's for the contrastive loss by passing through the projector head
            z_aug1 = self.projector(latent_reps_aug1)
            z_aug2 = self.projector(latent_reps_aug2)

            contrastive_loss = self.loss.compute(proj_feat0=None, proj_feat1=z_aug1, proj_feat2=z_aug2, partition_size=None, prediction=None)
            return contrastive_loss
        
        elif strategy == 'GD-' or strategy == 'GD-alt' or strategy == 'GD':
            orig_img, img_aug1, img_aug2 = batch    # size of each is batch_size, partition_num, 1, 192, 192 
            orig_img, img_aug1, img_aug2 = orig_img.float(), img_aug1.float(), img_aug2.float()
            b, p, h, w = orig_img.squeeze().shape

            # flattened so that batch_size and the partition_num are clubbed and the in_channels remain the same
            orig_img = orig_img.view(b*p, 1, h, w)  # so the batch_size dimension essentially becomes batch_size*partition
            img_aug1 = img_aug1.view(b*p, 1, h, w)
            img_aug2 = img_aug2.view(b*p, 1, h, w)

            # get the latent representations by passing each augmented image through the encoder
            latent_reps_unaug, context_feats_unaug = self.encoder(orig_img)
            latent_reps_aug1, context_feats_aug1 = self.encoder(img_aug1)
            latent_reps_aug2, context_feats_aug2 = self.encoder(img_aug2)
            # get the final z's for the contrastive loss by passing through the projector head
            z_aug0 = self.projector(latent_reps_unaug)
            z_aug1 = self.projector(latent_reps_aug1)
            z_aug2 = self.projector(latent_reps_aug2)

            contrastive_loss = self.loss.compute(proj_feat0=z_aug0, proj_feat1=z_aug1, proj_feat2=z_aug2, partition_size=p, prediction=None)

            return contrastive_loss

    def training_step(self, batch, batch_nb):
        if self.cfg.strategy == 'GR':
            imgs, _ = batch
        else:
            imgs, _, _ = batch
        loss = self.compute_loss(batch, strategy=self.cfg.strategy)
        self.train_losses += [loss.item()] * len(imgs)

        if loss < self.best_train_loss - self.loss_visualization_step and batch_nb==0:
          self.best_train_loss = loss.item()

        return loss

    def validation_step(self, batch, batch_nb):
        if self.cfg.strategy == 'GR':
            imgs, _ = batch
        else:
            imgs, _, _ = batch
        loss = self.compute_loss(batch, strategy=self.cfg.strategy)
        # self.log('valid_contrastive_loss', loss, on_step=False, on_epoch=True)
        self.valid_losses += [loss.item()] * len(imgs)

        # qualitative results on wandb when first batch dice improves by 10%
        if loss < self.best_valid_loss - self.loss_visualization_step and batch_nb==0:
          self.best_valid_loss = loss.item()
    
    def on_train_epoch_end(self):
        train_loss = np.mean(self.train_losses)
        self.log('train_loss', train_loss)
        self.train_losses = []

    def on_validation_epoch_end(self):
        valid_loss = np.mean(self.valid_losses)
        self.log('valid_loss', valid_loss)
        self.valid_losses = []

    def configure_optimizers(self):
        optimizer = optim.AdamW(params=self.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.cfg.T_0//self.num_iters_per_epoch, eta_min=1e-6)        
        return [optimizer], [scheduler]
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.cfg.batch_size,
                             shuffle = True, drop_last=True, num_workers=self.cfg.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size = self.cfg.batch_size,
                             shuffle = False, drop_last=False, num_workers=self.cfg.num_workers)


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
    fig.tight_layout()
    fig.show()
    return fig

def main(cfg):
    # experiment tracker (you need to sign in with your account)
    timestamp = time.time()
    wandb_logger = pl.loggers.WandbLogger(
                            name='%s - %s - %s <- %d'%(cfg.strategy, cfg.dataset_name, cfg.num_train_imgs, timestamp), 
                            group= '%s'%(cfg.exp_name), 
                            log_model=True, # save best model using checkpoint callback
                            project='self-supervised-pretrain',
                            entity='ssl-medical-imaging',
                            config=cfg,
    )

    # to save the best model on validation, log learning_rate and early stop
    checkpoint = pl.callbacks.ModelCheckpoint(
        filename="best_model_"+str(int(timestamp)),
        monitor="valid_loss", save_top_k=1, save_last=False, mode="min")
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    early_stop = pl.callbacks.EarlyStopping(monitor="valid_loss", min_delta=0.00,
                                            patience=cfg.patience, verbose=False, mode="min")

    model = SegModel(cfg)

    trainer = pl.Trainer(
        devices=cfg.num_gpus, accelerator="gpu", strategy="ddp",
        logger=wandb_logger, 
        callbacks=[checkpoint, lr_monitor, early_stop],
        max_epochs=cfg.epochs, 
        precision=cfg.precision,
        enable_progress_bar=cfg.enable_progress_bar,
        check_val_every_n_epoch=(cfg.val_every_n_iters//model.num_iters_per_epoch))

    # log gradients, parameter histogram and model topology
    wandb_logger.watch(model, log='all')
    
    trainer.fit(model)
    print("------- Training Done! -------")

    print("------- Loading the Best Model! ------")     # the PyTorch Lightning way
    # load the best checkpoint after training
    loaded_model = model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, strict=False)
    pretrained_encoder = loaded_model.encoder
    
    strategy = cfg.strategy     # options: "GR", "GD-", "GD", "GD-alt"
    folder_name = "./" + strategy + "_saved_models/"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    save_path = folder_name + "best_encoder_" + strategy + "_" + cfg.dataset + ".pt"   

    torch.save(pretrained_encoder.state_dict(), save_path)

    # print("------- Testing Begins! -------")
    # trainer.test(model)

if __name__ == '__main__':
    main(cfg)