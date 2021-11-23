# utility packages
import os
import time
# pip install python-box
from box import Box
import matplotlib.pyplot as plt
timestamp = time.time()

# machine learning packages
import wandb
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# list of configurables
cfg = {
        # summary
        'exp_name': 'Testing!',
        # dataset
        'dataset': ...,
        # model
        'model': 'Unet',
        'num_classes': ...,
        'precision': 32,
        # optimization
        'epochs': 10,
        'batch_size': 32,
        'opt': 'torch.optim.AdamW',
        'opt_params':{
            'lr': 1e-4,
            'weight_decay': 0,
        },
        'num_worker': 2,
        # scheduler
        'scheduler': 'torch.optim.lr_scheduler.CosineAnnealingWarmRestarts',
        'scheduler_params':{
            'T_0': 40,
            'eta_min': 1e-5
        },
        # hardware
        'num_gpus': 2,
    }
cfg = Box(cfg)


class SegModel(pl.LightningModule):
    def __init__(self, cfg):
        super(SegModel, self).__init__()
        self.cfg = cfg
        self.net = ...
        self.train_dataset = ...
        self.valid_dataset = ...
        self.test_dataset = ...
        
    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch, batch_nb):
        loss = ...
        return {'train_loss' : loss}

    def validation_step(self, batch, batch_nb):
        dice = ...
        return {'valid_dice' : dice}
    
    def test_step(self, batch, batch_nb):
        dice = ...
        return {'test_dice' : dice}

    def configure_optimizers(self):
        optimizer = eval(self.cfg.opt)(
            self.parameters(), **self.cfg.opt_params
        )
        scheduler = eval(self.cfg.scheduler)(
            optimizer,
            **self.cfg.scheduler_params
        )
        return [optimizer], [scheduler]
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.cfg.batch_size,
                             shuffle = True, drop_last=True, num_workers=self.cfg.num_worker)

    def valid_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size = self.cfg.batch_size,
                             shuffle = False, drop_last=False, num_workers=self.cfg.num_worker)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size = self.cfg.batch_size,
                             shuffle = False, drop_last=False, num_workers=self.cfg.num_worker)

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
    lr_monitor = pl.callbacks.LearningRateMonitor(
        logging_interval='epoch',
    )

    trainer = pl.Trainer(
        devices=cfg.num_gpus, accelerator="gpu", strategy="ddp",
        logger=wandb_logger,
        callbacks=[checkpoint, lr_monitor],
        max_epochs=cfg.epochs,
        precision=cfg.precision
    )

    model = SegModel(cfg)
    trainer.fit(model)

if __name__ == '__main__':
    main(cfg)