# utility packages
import os
import time
import argparse

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
from seg_models_v2 import UNetEncoder, UNetDecoder
from Dataset.init_data import acdc, md_prostate
from Dataset.dataset import DatasetRandom       # note: for finetuning we always use DatasetRandom
from Dataset.experiments_paper import data_init_acdc, data_init_prostate_md
from loss import Loss, multiclass_dice_coeff

# define paths for the images and the segmentation labels
img_path = "/home/ssl_project/datasets/ACDC"
seg_path = "/home/ssl_project/datasets/ACDC"

parser = argparse.ArgumentParser(description="Random-Random Strategy Run 3")

# all the arguments for the dataset, model, and training hyperparameters
parser.add_argument('--exp_name', default='FT FINAL Final', type=str, help='Name of the experiment/run')
parser.add_argument('-st', '--strategy', default='GR', type=str, help='Strategy for pretraining; Options: GR, GD-, GD, GD-alt')
# dataset
parser.add_argument('-data', '--dataset', default='ACDC', help='Specifyg acdc or md_prostate without quotes')
parser.add_argument('--dataset_name', default='acdc', type=str, help='acdc or md_prostate dataset')
parser.add_argument('-nti', '--num_train_imgs', default='tr8', type=str, help='Number of training images, options tr1, tr2 or tr8')
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
parser.add_argument('-ep', '--epochs', default=2500, type=int, help='Number of epochs to train')
parser.add_argument('-bs', '--batch_size', default=12, type=int, help='Batch size')
parser.add_argument('-nw', '--num_workers', default=4, type=int, help='Number of worker processes')
parser.add_argument('-gpus', '--num_gpus', default=1, type=int, help="Number of GPUs to use")
parser.add_argument('-lr', '--learning_rate', default=5e-4, type=float, help="Learning rate to use")
parser.add_argument('-wd', '--weight_decay', default=1e-3, type=float, help='Default weight decay')
parser.add_argument('-pat', '--patience', default=10, type=int, help='number of validation steps (val_every_n_iters) to wait before early stoping')
parser.add_argument('--T_0', default=500, type=int, help='number of steps in each cosine cycle')
parser.add_argument('-epb', '--enable_progress_bar', default=False, type=bool, help='by default is disabled since it doesnt work in colab')
parser.add_argument('--val_every_n_iters', default='100', type=int, help='num of iterations before validation')

parser.add_argument('-ptr', '--pretrained', default=False, action='store_true', help='Load pretrained weights for the encoder')

cfg = parser.parse_args()

# load paths for different pre-trained encoders
strategy = cfg.strategy     # options: "GR", "GD-", "GD", "GD-alt"
folder_name = "./" + strategy + "_saved_models/"
load_path = folder_name + "best_encoder_" + cfg.strategy + "_" + cfg.dataset + ".pt"
print(load_path)


class SegModel(pl.LightningModule):
    def __init__(self, cfg):
        super(SegModel, self).__init__()
        self.cfg = cfg
        if not cfg.pretrained:
            print("INITIALIZING ENCODER WEIGHTS FROM SCRATCH!")
            # self.net = UNet(n_channels=self.cfg.in_channels, init_filters=self.cfg.init_num_filters, n_classes=self.cfg.num_classes)
            self.encoder = UNetEncoder(n_channels=self.cfg.in_channels, init_filters=self.cfg.init_num_filters)
            self.decoder = UNetDecoder(init_filters=self.cfg.init_num_filters, n_classes=self.cfg.num_classes)
        
        else:
            print("LOADING PRETRAINED WEIGHTS FOR THE ENCODER!")
            self.encoder = UNetEncoder(n_channels=self.cfg.in_channels, init_filters=self.cfg.init_num_filters)
            self.encoder.load_state_dict(torch.load(load_path))
            self.encoder.eval()     # layers are frozen by using .eval() method
            # for finetuning - all the paramters are updated after initialization with pretrained weights
            # for params in self.encoder.parameters():
            #     params.requires_grad = False
            self.decoder = UNetDecoder(init_filters=self.cfg.init_num_filters, n_classes=self.cfg.num_classes)

        if cfg.dataset == 'ACDC':
          data_init = data_init_acdc
          dataset_cfg = acdc 
        elif cfg.dataset  == 'MD_PROSTATE':
          data_init = data_init_prostate_md
          dataset_cfg = md_prostate
        else:
          print('The dataset is not found')

        self.num_class = dataset_cfg['num_class']
        self.train_ids = data_init.train_data(self.cfg.num_train_imgs, self.cfg.comb_train_imgs)
        self.val_ids = data_init.val_data(self.cfg.num_train_imgs, self.cfg.comb_train_imgs)
        self.test_ids = data_init.test_data()

        self.train_dataset = DatasetRandom(dataset_cfg, self.train_ids, self.cfg.img_path, preprocessed_data=False, seg_path=self.cfg.seg_path, augmentation=True)
        self.valid_dataset = DatasetRandom(dataset_cfg, self.val_ids, self.cfg.img_path, preprocessed_data=False, seg_path=self.cfg.seg_path, augmentation=True)
        self.test_dataset = DatasetRandom(dataset_cfg, self.test_ids, self.cfg.img_path, preprocessed_data=False, seg_path=self.cfg.seg_path, augmentation=False)

        self.loss = Loss(loss_type=0, device=self.device)

        self.loss_visualization_step = 0.1
        self.best_valid_loss = 1
        self.best_train_loss = 1

        self.train_losses, self.valid_losses, self.test_losses = [], [], []
        self.init_timestamp = time.time()
        self.num_iters_per_epoch = np.int(np.ceil(len(self.train_dataset) / self.cfg.batch_size))
        
    def forward(self, x):
        enc_out, context_feats = self.encoder(x)
        logits, out_final = self.decoder(enc_out, context_feats)
        return out_final

    def compute_loss(self, batch, exclude_background=False):
        imgs, gts = batch
        imgs, gts = imgs.float(), gts.long()
        enc_out, context_feats = self.encoder(imgs)
        logits, preds = self.decoder(enc_out, context_feats)

        gts_one_hot = self.loss.one_hot(gts, num_classes=self.num_class)  # convert to one-hot for Dice loss
        loss = self.loss.compute(proj_feat0=None, proj_feat1=None, proj_feat2=None, partition_size=None, 
                                    prediction=preds[:, int(exclude_background):],target=gts_one_hot[:, int(exclude_background):], multiclass=True)
        return loss, preds, imgs, gts        
    
    def training_step(self, batch, batch_nb):
        loss, preds, imgs, gts = self.compute_loss(batch)
        self.train_losses += [loss.item()] * len(imgs)

        if loss < self.best_train_loss - self.loss_visualization_step and batch_nb==0:
            self.best_train_loss = loss.item()
            fig = visualize(preds, imgs, gts)
            wandb.log({"Training Output Visualizations": fig})
            plt.close()
        return loss

    def validation_step(self, batch, batch_nb):
        loss, preds, imgs, gts = self.compute_loss(batch)
        self.valid_losses += [loss.item()] * len(imgs)

        # qualitative results on wandb when first batch dice improves by 10%
        if loss < self.best_valid_loss - self.loss_visualization_step and batch_nb==0:
            self.best_valid_loss = loss.item()
            fig = visualize(preds, imgs, gts)
            wandb.log({"Validation Output Visualizations": fig})
            plt.close()
    
    def test_step(self, batch, batch_nb):
        loss, preds, imgs, gts = self.compute_loss(batch, exclude_background=True)
        self.test_losses += [loss.item()] * len(imgs)

        # qualitative results on wandb
        fig = visualize(preds, imgs, gts)
        wandb.log({"Test Output Visualizations": fig})
        plt.close()

        # # dice score (found online)
        # test_gt_one_hot = self.loss.one_hot(gts.long(), num_classes=self.cfg.num_classes)
        # # computing the dice score, ignoring the background ie class 0  
        # # but the paper does it with including the background, hence also adding class 0
        # test_dice_score = multiclass_dice_coeff((preds[:, 0:, ...]).to(self.device), (test_gt_one_hot[:, 0:, ...]).to(self.device), reduce_batch_first=True)   
        # # as a sanity check, dice score increases when background is also included (i.e. :, 0:, :, :)
        # outputs = dict({'test_dice_score' : test_dice_score})        
        # return outputs
    
    def on_train_epoch_end(self):
        train_loss = np.mean(self.train_losses)
        self.log('train_loss', train_loss)
        self.train_losses = []

    def on_validation_epoch_end(self):
        valid_loss = np.mean(self.valid_losses)
        self.log('valid_loss', valid_loss)
        self.valid_losses = []

    def on_test_epoch_end(self):
        test_loss = np.mean(self.test_losses)
        self.log('test_loss', test_loss)
        self.test_losses = []
        self.final_timestamp = time.time()
        self.log('train+test_time', self.final_timestamp - self.init_timestamp)

    # def test_epoch_end(self, outputs):
    #     temp = torch.stack([x['test_dice_score'] for x in outputs])
    #     # print(temp)
    #     mean_dice = temp.mean()
    #     print(f"MEAN DICE SCORE: {mean_dice}")
            
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
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size = self.cfg.batch_size,
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
    fig.show()
    return fig

def main(cfg):
    # experiment tracker (you need to sign in with your account)
    timestamp = time.time()
    wandb_logger = pl.loggers.WandbLogger(
                            name='%s-%s-%s-%s <- %d'%(cfg.strategy, cfg.dataset_name, cfg.num_train_imgs, cfg.comb_train_imgs, timestamp), 
                            group= '%s %s'%(cfg.exp_name, cfg.strategy), 
                            log_model=True, # save best model using checkpoint callback
                            project='supervised-finetune',
                            entity='ssl-medical-imaging',
                            config=cfg)

    # to save the best model on validation, log learning_rate and early stop
    checkpoint = pl.callbacks.ModelCheckpoint(
        filename="best_model_"+str(int(timestamp)),
        monitor="valid_loss", 
        save_top_k=1, 
        save_last=False, 
        mode="min")
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

    print("------- Testing Begins! -------")
    trainer.test(model)

if __name__ == '__main__':
    main(cfg)