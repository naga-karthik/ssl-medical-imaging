# utility packages
import argparse
import time
import os

# uncomment for local run
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#os.environ["WANDB_API_KEY"] = "c2afd1f40f749fb27430c7ed36a6f3f2d425e6dc"

timestamp = time.time()

# machine learning packages
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# dataloaders and segmentation models
from seg_models_v2 import UNetEncoder, ProjectorHead
from Dataset.init_data import acdc
from Dataset.dataset import DatasetGR
from Dataset.experiments_paper import data_init_acdc
from loss import Loss

#img_path = "ACDC" # for local run
img_path = "/home/GRAMES.POLYMTL.CA/u114716/ssl_project/datasets/ACDC"
seg_path = None

parser = argparse.ArgumentParser(description="gl-GR-Random Strategy Run 1")

# all the arguments for the dataset, model, and training hyperparameters
parser.add_argument('--exp_name', default='gl-GR-R_Test-SSL', type=str, help='Name of the experiment/run')
# dataset
parser.add_argument('-data', '--dataset', default=acdc, help='Specifyg acdc or md_prostate without quotes')
parser.add_argument('-partition', '--partition', default=4, help='Specifyg number of partitions')
parser.add_argument('-nti', '--num_train_imgs', default='tr52', type=str, help='Number of training images, options tr1, tr8 or tr52')
parser.add_argument('-cti', '--comb_train_imgs', default='c1', type=str, help='Combintation of Train imgs., options c1, c2, cr3, cr4, cr5')
parser.add_argument('--img_path', default=img_path, type=str, help='Absolute path of the training data')
parser.add_argument('--seg_path', default=seg_path, type=str, help='Same as path of training data')
# model
parser.add_argument('-in_ch', '--in_channels', default=1, type=int, help='Number of input channels')
parser.add_argument('-num_flt', '--init_filters', type=int, default=16, help='Initial no. of filters for Conv Layers')
parser.add_argument('-num_flt_enc', '--encoder_init_filters', type=int, default=16, help='Initial no. of filters for encoder')
parser.add_argument('-num_fc', '--fc_units_list', nargs='+', default=[3200, 1024], help='List containing no. of units in FC layers')
parser.add_argument('-g1_dim', '--out_dim', default=128, type=int, help='Output dimension for the projector head')
# optimization
parser.add_argument('-p', '--precision', default=32, type=int, help='Precision for training')
parser.add_argument('-ep', '--epochs', default=100, type=int, help='Number of epochs to train')
parser.add_argument('-bs', '--batch_size', default=64, type=int, help='Batch size')
parser.add_argument('-nw', '--num_workers', default=4, type=int, help='Number of worker processes')
parser.add_argument('-gpus', '--num_gpus', default=1, type=int, help="Number of GPUs to use")
parser.add_argument('-lr', '--learning_rate', default=1e-3, type=float, help="Learning rate to use")
parser.add_argument('-wd', '--weight_decay', default=0.0, type=float, help='Default weight decay')

cfg = parser.parse_args()

class EncoderPretrain(pl.LightningModule):
    def __init__(self, cfg):
        super(EncoderPretrain, self).__init__()
        self.cfg = cfg
        self.e = UNetEncoder(n_channels=self.cfg.in_channels, init_filters=self.cfg.init_filters)
        self.g1 = ProjectorHead(encoder_init_filters=self.cfg.encoder_init_filters, out_dim=self.cfg.out_dim)

        self.train_ids_acdc = data_init_acdc.train_data(self.cfg.num_train_imgs, self.cfg.comb_train_imgs)
        self.val_ids_acdc = data_init_acdc.val_data(self.cfg.num_train_imgs, self.cfg.comb_train_imgs)
        self.test_ids_acdc = data_init_acdc.test_data()

        self.train_dataset = DatasetGR(self.cfg.dataset, self.train_ids_acdc, self.cfg.img_path, preprocessed_data=True, seg_path=None)
        self.valid_dataset = DatasetGR(self.cfg.dataset, self.val_ids_acdc, self.cfg.img_path, preprocessed_data=True, seg_path=None)
        self.test_dataset = DatasetGR(self.cfg.dataset, self.test_ids_acdc, self.cfg.img_path, preprocessed_data=True, seg_path=None)

        self.loss = Loss(loss_type=1, encoder_strategy=1, device=self.device)
        
    def forward(self, x):
        e_out = self.e(x)
        g1_out = self.g1(e_out)
        return g1_out
    
    def training_step(self, batch, batch_nb):
        train_aug1, train_aug2 = batch
        train_aug1, train_aug2 = train_aug1.float(), train_aug2.float()

        encoder_out_aug1 = self.e(train_aug1)  # self(train_img)
        out_aug1 = self.g1(encoder_out_aug1)
        encoder_out_aug2 = self.e(train_aug2)  # self(train_img)
        out_aug2 = self.g1(encoder_out_aug2)

        # Gr loss
        loss = self.loss.compute(out_aug1, out_aug2)
        self.log('train_loss', loss, on_step=False, on_epoch=True)

        return {'loss' : loss}

    def validation_step(self, batch, batch_nb):
        val_aug1, val_aug2 = batch
        val_aug1, val_aug2 = val_aug1.float(), val_aug2.float()

        encoder_out_aug1 = self.e(val_aug1)  # self(train_img)
        out_aug1 = self.g1(encoder_out_aug1)
        encoder_out_aug2 = self.e(val_aug2)  # self(train_img)
        out_aug2 = self.g1(encoder_out_aug2)

        # Gr loss
        loss = self.loss.compute(out_aug1, out_aug2)
        self.log('valid_loss', loss, on_step=False, on_epoch=True)

        return {'loss': loss}

    # removed test step

    def configure_optimizers(self):
        optimizer = optim.Adam(params=self.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=40, eta_min=1e-5)
        
        return [optimizer], [scheduler]

    def train_dataloader(self):
        data = DataLoader(self.train_dataset, batch_size = self.cfg.batch_size,
                             shuffle = True, drop_last=True, num_workers=self.cfg.num_workers)
        return data

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size = self.cfg.batch_size,
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
        filename="best_model_encoder_pretrain"+str(timestamp),
        monitor="valid_loss",
        save_top_k=1,
        mode="max",
        save_last=False,
        save_weights_only=True,
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        devices=cfg.num_gpus, accelerator="gpu", 
        strategy="ddp",
        logger=wandb_logger,
        callbacks=[checkpoint, lr_monitor],
        max_epochs=cfg.epochs,
        precision=cfg.precision,
    )

    model = EncoderPretrain(cfg)
    # log gradients, parameter histogram and model topology
    wandb_logger.watch(model, log='all')
    
    trainer.fit(model)
    print("------- PreTraining Done! -------")


if __name__ == '__main__':
    main(cfg)