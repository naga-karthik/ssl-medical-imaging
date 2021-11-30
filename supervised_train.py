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

parser = argparse.ArgumentParser(description="Random-Random Strategy Run 3")

# all the arguments for the dataset, model, and training hyperparameters
parser.add_argument('--exp_name', default='R-R_epch-100_bs-64_dice_w-test', type=str, help='Name of the experiment/run')
# dataset
parser.add_argument('-data', '--dataset', default=acdc, help='Specifyg acdc or md_prostate without quotes')
parser.add_argument('-nti', '--num_train_imgs', default='tr52', type=str, help='Number of training images, options tr1, tr8 or tr52')
parser.add_argument('-cti', '--comb_train_imgs', default='c1', type=str, help='Combintation of Train imgs., options c1, c2, cr3, cr4, cr5')
parser.add_argument('--img_path', default=img_path, type=str, help='Absolute path of the training data')
parser.add_argument('--seg_path', default=seg_path, type=str, help='Same as path of training data')
# model
parser.add_argument('-in_ch', '--in_channels', default=1, type=int, help='Number of input channels')
parser.add_argument('-num_flt', '--num_filters_list', nargs='+', default=[1, 16, 32, 64, 128, 256], help='List containing no. of filters for Conv Layers')
parser.add_argument('-num_fc', '--fc_units_list', nargs='+', default=[3200, 1024], help='List containing no. of units in FC layers')
parser.add_argument('-g1_dim', '--g1_out_dim', default=128, type=int, help='Output dimension for the projector head')
parser.add_argument('-nc', '--num_classes', default=4, type=int, help='Number of classes to segment')
# optimization
parser.add_argument('-p', '--precision', default=32, type=int, help='Precision for training')
parser.add_argument('-ep', '--epochs', default=100, type=int, help='Number of epochs to train')
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

        self.train_dataset = DataloaderRandom(self.cfg.dataset, self.train_ids_acdc, self.cfg.img_path, preprocessed_data=True, seg_path=self.cfg.seg_path)
        self.valid_dataset = DataloaderRandom(self.cfg.dataset, self.val_ids_acdc, self.cfg.img_path, preprocessed_data=True, seg_path=self.cfg.seg_path)
        self.test_dataset = DataloaderRandom(self.cfg.dataset, self.test_ids_acdc, self.cfg.img_path, preprocessed_data=True, seg_path=self.cfg.seg_path)

        self.loss = Loss(loss_type=0, device=self.device)
        
    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch, batch_nb):
        train_img, train_gt = batch
        train_img = train_img.float()
        train_gt = train_gt.long()    #train_gt.float()
        
        logits, out_final = self.net(train_img)  # self(train_img)
        # dice loss
        loss = self.loss.compute(out_final, train_gt)
        self.log('train_dice_loss', loss)
        # # cross-entropy loss
        # loss = F.cross_entropy(logits, train_gt.squeeze(dim=1))
        # self.log('train_ce_loss', loss)
        return {'loss' : loss}

    def validation_step(self, batch, batch_nb):
        print(len(batch))
        val_img, val_gt = batch
        val_img = val_img.float() #.to(self.device)
        val_gt = val_gt.long() # val_gt.float()   #.to(self.device)
        
        val_logits, val_out_final = self.net(val_img)    # self(val_img)
        # dice loss        
        loss = self.loss.compute(val_out_final, val_gt)
        self.log('valid_loss', loss)
        # # cross-entropy loss
        # loss = F.cross_entropy(val_logits, val_gt.squeeze(dim=1))
        # self.log('valid_loss', loss)
        return {'loss' : loss}
    
    def test_step(self, batch, batch_nb):
        test_img, test_gt = batch
        test_img = test_img.float()

        _, y_hat = self.forward(test_img)

        test_dice_loss = self.loss.compute(y_hat, test_gt.long())
        test_dice_score = (1.0 - test_dice_loss)    # recall that we were return 1-dice_loss for PyTorch to minimize the loss
        print(f"\nDICE SCORE ON THE TEST SET: {test_dice_score}")

        # # print the shapes of all tensors first
        # print(f"test image shape: {test_img.shape}")    # shape: [batch_size, 1, 192, 192]
        # print(f"test gt shape: {test_gt.shape}")        # shape: [batch_size, 1, 192, 192]
        # print(f"prediction shape: {y_hat.shape}")       # shape: [batch_size, num_classes, 192, 192]

        if batch_nb % 20 == 0:
            # plot every 20th image
            test_img, test_gt = test_img.squeeze(dim=1).cpu().numpy(), test_gt.squeeze(dim=1).cpu().numpy()
            y_hat_npy = y_hat.cpu().numpy()
            class_num = 1
            pred_img = y_hat_npy[:, class_num, :, :]

            # plot images on wandb
            fig, axs = plt.subplots(5, 4, figsize=(10, 10))
            fig.suptitle('Original --> Ground Truth --> Prediction --> Pred with Mask layover')
            for i in range(5):
                img_num = np.random.randint(0, self.cfg.batch_size)
                axs[i, 0].imshow(test_img[img_num], cmap='gray'); axs[i, 0].axis('off')
                axs[i, 1].imshow(test_gt[img_num], cmap='gray'); axs[i, 1].axis('off')    
                axs[i, 2].imshow(pred_img[img_num], cmap='gray'); axs[i, 2].axis('off')
                axs[i, 3].imshow((test_img[img_num] - pred_img[img_num]), cmap='gray'); axs[i, 3].axis('off')
            fig.show()
            wandb.log({"Output Visualizations": fig})

            # wandb.log({"Original image": [wandb.Image(test_img[5], caption=f"Original Test Image")]})
            # wandb.log({"Ground Truth": [wandb.Image(test_gt[5], caption=f"Ground Truth")]})
            # wandb.log({"prediction": [wandb.Image(pred_img[5], caption=f"Prediction Class {class_num}")]})
        
        return {'test_dice_score' : test_dice_score}

    def configure_optimizers(self):
        # opt_params = { 'lr': 1e-4, 'weight_decay': 0, }
        # scheduler_params={ 'T_0': 40, 'eta_min': 1e-5 }
        # optimizer = eval(self.cfg.opt)(self.parameters(), **self.cfg.opt_params)
        # scheduler = eval(self.cfg.scheduler)(optimizer, **self.cfg.scheduler_params)

        optimizer = optim.Adam(params=self.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
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
        monitor="valid_loss",
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

    print("------- Testing Begins! -------")
    trainer.test(model)

if __name__ == '__main__':
    main(cfg)