# utility packages
import os
import time
import argparse
from torch._C import device

from torch.nn.modules.module import T
import matplotlib.pyplot as plt
timestamp = time.time()

# machine learning packages
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# dataloaders and segmentation models
from seg_models import SegUnetFullModel, SegUnetEncoder_and_ProjectorG1, SegUnetDecoder
from Dataloader.init_data import acdc, md_prostate
from Dataloader.dataloader import DataloaderRandom
from Dataloader.experiments_paper import data_init_acdc, data_init_prostate_md
from loss import Loss

img_path = "/home/GRAMES.POLYMTL.CA/u114716/ssl_project/datasets/ACDC"
seg_path = "/home/GRAMES.POLYMTL.CA/u114716/ssl_project/datasets/ACDC"

parser = argparse.ArgumentParser(description="Random-Random Strategy Run 1")

# all the arguments for the dataset, model, and training hyperparameters
parser.add_argument('--exp_name', default='Random-Random Strategy Run 1', type=str, help='Name of the experiment/run')
# dataset
parser.add_argument('-data', '--dataset', default=acdc, help='Specifyg acdc or md_prostate without quotes')
parser.add_argument('-nti', '--num_train_imgs', default='tr52', type=str, help='Number of training images, options tr1, tr8 or tr52')
parser.add_argument('-cti', '--comb_train_imgs', default='c1', type=str, help='Combintation of Train imgs., options c1, c2, cr3, cr4, cr5')
parser.add_argument('--img_path', default=img_path, type=str, help='Absolute path of the training data')
parser.add_argument('--seg_path', default=seg_path, type=str, help='Same as path of training data')
# model
parser.add_argument('-in_ch', '--in_channels', default=1, type=int, help='Number of input channels')
parser.add_argument('-num_flt', '--num_filters_list', nargs='+', default=[1, 16, 32, 64, 128, 128], help='List containing no. of filters for Conv Layers')
parser.add_argument('-num_fc', '--fc_units_list', nargs='+', default=[3200, 1024], help='List containing no. of units in FC layers')
parser.add_argument('-g1_dim', '--g1_out_dim', default=128, type=int, help='Output dimension for the projector head')
parser.add_argument('-nc', '--num_classes', default=4, type=int, help='Number of classes to segment')
# optimization
parser.add_argument('-p', '--precision', default=32, type=int, help='Precision for training')
parser.add_argument('-ep', '--epochs', default=10, type=int, help='Number of epochs to train')
parser.add_argument('-bs', '--batch_size', default=32, type=int, help='Batch size')
parser.add_argument('-nw', '--num_workers', default=4, type=int, help='Number of worker processes')
parser.add_argument('-gpus', '--num_gpus', default=1, type=int, help="Number of GPUs to use")
parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float, help="Learning rate to use")
parser.add_argument('-wd', '--weight_decay', default=0.0, type=float, help='Default weight decay')

cfg = parser.parse_args()


class SegModel(pl.LightningModule):
    def __init__(self, cfg):
        super(SegModel, self).__init__()
        self.cfg = cfg
        self.net = SegUnetFullModel(
            in_channels=self.cfg.in_channels, 
            num_filters_list=self.cfg.num_filters_list,
            fc_units=self.cfg.fc_units_list,
            g1_out_dim=self.cfg.g1_out_dim, 
            num_classes=self.cfg.num_classes
        )

        self.train_ids_acdc = data_init_acdc.train_data(self.cfg.num_train_imgs, self.cfg.comb_train_imgs)
        self.val_ids_acdc = data_init_acdc.val_data(self.cfg.num_train_imgs, self.cfg.comb_train_imgs)
        self.test_ids_acdc = data_init_acdc.test_data()

        self.train_dataset = DataloaderRandom(self.cfg.dataset, self.train_ids_acdc, self.cfg.img_path, preprocessed_data=False, seg_path=self.cfg.seg_path)
        self.valid_dataset = DataloaderRandom(self.cfg.dataset, self.val_ids_acdc, self.cfg.img_path, preprocessed_data=False, seg_path=self.cfg.seg_path)
        self.test_dataset = DataloaderRandom(self.cfg.dataset, self.test_ids_acdc, self.cfg.img_path, preprocessed_data=False, seg_path=self.cfg.seg_path)

        self.loss = Loss(loss_type=0, device=self.device)
        
    def forward(self, x):
        # return self.net(x)
        pass
    
    def training_step(self, batch, batch_nb):
        train_img, train_gt = batch
        train_img = train_img.float()
        train_gt = train_gt.float()
        
        logits, out_final = self.net(train_img)  # self(train_img)
        train_dice_loss = self.loss.compute(out_final, train_gt)
        return {'loss' : train_dice_loss}

    def validation_step(self, batch, batch_nb):
        print(len(batch))
        val_img, val_gt = batch
        val_img = val_img.float() #.to(self.device)
        val_gt = val_gt.float()   #.to(self.device)
        
        _, val_out_final = self.net(val_img)    # self(val_img)
        valid_dice_loss = self.loss.compute(val_out_final, val_gt)
        return {'loss' : valid_dice_loss}
    
    # def test_step(self, batch, batch_nb):
    #     test_dice_loss = ...
    #     return {'test_dice' : test_dice_loss}

    def configure_optimizers(self):
        # opt_params = { 'lr': 1e-4, 'weight_decay': 0, }
        # scheduler_params={ 'T_0': 40, 'eta_min': 1e-5 }
        # optimizer = eval(self.cfg.opt)(self.parameters(), **self.cfg.opt_params)
        # scheduler = eval(self.cfg.scheduler)(optimizer, **self.cfg.scheduler_params)

        optimizer = optim.AdamW(params=self.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=40, eta_min=1e-5)
        
        return [optimizer], [scheduler]
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.cfg.batch_size,
                             shuffle = True, drop_last=True, num_workers=self.cfg.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size = self.cfg.batch_size,
                             shuffle = False, drop_last=False, num_workers=self.cfg.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size = self.cfg.batch_size,
                             shuffle = False, drop_last=False, num_workers=self.cfg.num_workers)

def main(cfg):
    # experiment tracker (you need to sign in with your account)
    wandb_logger = pl.loggers.WandbLogger(
                            name='%s <- %d'%(cfg.exp_name, timestamp), 
                            group= '%s'%(cfg.exp_name), 
                            log_model=True, # save best model using checkpoint callback
                            project='supervised-train',
                            entity='ssl-medical-imaging',
                            config=cfg,
    )

    # to save the best model on validation
    checkpoint = pl.callbacks.ModelCheckpoint(
        filename="best_model"+str(timestamp),
        monitor="valid_dice",
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
    trainer.fit(model)

if __name__ == '__main__':
    main(cfg)