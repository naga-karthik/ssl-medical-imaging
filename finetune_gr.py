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
import argparse
import time
import os

# uncomment for local run
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["WANDB_API_KEY"] = "c2afd1f40f749fb27430c7ed36a6f3f2d425e6dc"

timestamp = time.time()

# machine learning packages
import torch
import torch.optim as optim
import pytorch_lightning as pl
import wandb
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, dataloader
import torchvision.transforms as transforms

# dataloaders and segmentation models

from seg_models_v2 import UNetEncoder, UNetDecoder
from Dataset.init_data import acdc
from Dataset.dataset import DatasetRandom
from Dataset.experiments_paper import data_init_acdc
from loss import Loss, multiclass_dice_coeff

img_path = "ACDC"
seg_path = "ACDC"
load_path = "./pretrained_encoder.pt"

parser = argparse.ArgumentParser(description="gl-Gr-Random-finetune")

# all the arguments for the dataset, model, and training hyperparameters
parser.add_argument('--exp_name', default='gl-GR-R_Finetune-tr8', type=str, help='Name of the experiment/run')
# dataset
parser.add_argument('-data', '--dataset', default=acdc, help='Specifyg acdc or md_prostate without quotes')
parser.add_argument('-partition', '--partition', default=4, help='Specifyg number of partitions')
parser.add_argument('-nti', '--num_train_imgs', default='tr52', type=str,
                    help='Number of training images, options tr1, tr8 or tr52')
parser.add_argument('-cti', '--comb_train_imgs', default='c1', type=str,
                    help='Combintation of Train imgs., options c1, c2, cr3, cr4, cr5')
parser.add_argument('--img_path', default=img_path, type=str, help='Absolute path of the training data')
parser.add_argument('--seg_path', default=seg_path, type=str, help='Same as path of training data')
# model
parser.add_argument('-in_ch', '--in_channels', default=1, type=int, help='Number of input channels')
parser.add_argument('-num_flt', '--init_filters', type=int, default=16, help='Initial no. of filters for Conv Layers')
parser.add_argument('-num_flt_enc', '--encoder_init_filters', type=int, default=16,
                    help='Initial no. of filters for encoder')
parser.add_argument('-num_fc', '--fc_units_list', nargs='+', default=[3200, 1024],
                    help='List containing no. of units in FC layers')
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
        self.encoder = UNetEncoder(n_channels=self.cfg.in_channels, init_filters=self.cfg.init_filters)
        self.encoder.load_state_dict(torch.load(load_path))
        self.encoder.eval()  # layers are frozen by using .eval() method
        for params in self.encoder.parameters():
            params.requires_grad = False

        self.decoder = UNetDecoder(init_filters=self.cfg.init_filters, n_classes=self.cfg.num_classes)

        self.train_ids_acdc = data_init_acdc.train_data(self.cfg.num_train_imgs, self.cfg.comb_train_imgs)
        self.val_ids_acdc = data_init_acdc.val_data(self.cfg.num_train_imgs, self.cfg.comb_train_imgs)
        self.test_ids_acdc = data_init_acdc.test_data()

        self.train_dataset = DatasetRandom(self.cfg.dataset, self.train_ids_acdc, self.cfg.img_path,
                                           preprocessed_data=True, seg_path=self.cfg.seg_path, augmentation=True)
        self.valid_dataset = DatasetRandom(self.cfg.dataset, self.val_ids_acdc, self.cfg.img_path,
                                           preprocessed_data=True, seg_path=self.cfg.seg_path, augmentation=True)
        self.test_dataset = DatasetRandom(self.cfg.dataset, self.test_ids_acdc, self.cfg.img_path,
                                          preprocessed_data=True, seg_path=self.cfg.seg_path, augmentation=False)

        self.loss = Loss(loss_type=0, device=self.device)

    def forward(self, x):
        enc_out, context_feats = self.encoder(x)
        logits, out_final = self.decoder(enc_out, context_feats)
        return out_final

    def compute_loss(self, batch):
        imgs, gts = batch
        imgs, gts = imgs.float(), gts.long()
        enc_out, context_feats = self.encoder(imgs)
        logits, preds = self.decoder(enc_out, context_feats)
        # print(torch.unique(gts), gts.shape)

        gts_one_hot = self.loss.one_hot(gts, num_classes=self.cfg.num_classes)  # convert to one-hot for Dice loss
        loss = self.loss.compute(preds, aug2=None, unaug=None, target=gts_one_hot, multiclass=True)
        return loss, preds, imgs, gts

    def training_step(self, batch, batch_nb):
        loss, preds, imgs, gts = self.compute_loss(batch)
        self.log('train_loss', loss, on_step=False, on_epoch=True)

        if batch_nb == 0:  # once per epoch
            fig = visualize(preds, imgs, gts)
            wandb.log({"Training Output Visualizations": fig})
        #return loss
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        loss, preds, imgs, gts = self.compute_loss(batch)
        self.log('valid_loss', loss, on_step=False, on_epoch=True)

        if batch_nb == 0:  # once per epoch
            fig = visualize(preds, imgs, gts)
            wandb.log({"Validation Output Visualizations": fig})
        return {'loss': loss}

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

        return {'test_dice_score': test_dice_score}
    
    def configure_optimizers(self):
        optimizer = optim.Adam(params=self.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=40, eta_min=1e-5)

        return [optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.cfg.batch_size,
                          shuffle=True, drop_last=True, num_workers=self.cfg.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.cfg.batch_size,
                          shuffle=False, drop_last=False, num_workers=self.cfg.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.cfg.batch_size,
                          shuffle=False, drop_last=False, num_workers=self.cfg.num_workers)

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
        axs[0, i].imshow(imgs[img_num], cmap='gray');
        axs[0, i].axis('off')
        axs[1, i].imshow(gts_colored[img_num]);
        axs[1, i].axis('off')
        axs[2, i].imshow(preds_colored[img_num]);
        axs[2, i].axis('off')
    fig.show()
    return fig

def main(cfg):
    # experiment tracker (you need to sign in with your account)
    wandb_logger = pl.loggers.WandbLogger(
        name='%s <- %d' % (cfg.exp_name, timestamp),
        group='%s' % (cfg.exp_name),
        log_model=True,  # save best model using checkpoint callback
        project='supervised-finetune',
        entity='ssl-medical-imaging',
        config=cfg,
    )

    # to save the best model on validation
    checkpoint = pl.callbacks.ModelCheckpoint(
        filename="best_model" + str(timestamp),
        monitor="valid_loss",
        save_top_k=1,
        mode="max",
        save_last=False,
        save_weights_only=True,
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        devices=cfg.num_gpus, accelerator="gpu",
        #strategy="ddp",
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
    torch.save(model.state_dict(), 'finetune-gr.pt')

    print("------- Testing Begins! -------")
    trainer.test(model)

#-------------------------
    # experiment tracker (you need to sign in with your account)
    wandb_logger = pl.loggers.WandbLogger(
        name='%s <- %d' % (cfg.exp_name, timestamp),
        group='%s' % (cfg.exp_name),
        log_model=True,  # save best model using checkpoint callback
        project='supervised-train',
        entity='ssl-medical-imaging',
        config=cfg,
    )

    # to save the best model on validation
    checkpoint = pl.callbacks.ModelCheckpoint(
        filename="best_model_encoder_pretrain" + str(timestamp),
        monitor="valid_loss",
        save_top_k=1,
        mode="max",
        save_last=False,
        save_weights_only=True,
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        devices=cfg.num_gpus, accelerator="gpu",
        # strategy="ddp",
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
    torch.save(model.state_dict(), 'finetune-gr.pt')

    print("------- Testing Begins! -------")
    trainer.test(model)

if __name__ == '__main__':
    main(cfg)