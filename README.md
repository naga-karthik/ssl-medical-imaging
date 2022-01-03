# ssl-medical-imaging

This repo contains the PyTorch-Lightning reimplementation of the paper "[Contrastive learning of global and local features for medical image segmentation with limited annotations](https://arxiv.org/pdf/2006.10511.pdf)" by Chaitanya et al. (2020). Particularly, the experiments in the paper involving the Global Contrastive Loss and all the self-supervised pre-training strategies associated with it are reimplemented. The experiment tracking is done using [WandB](https://wandb.ai). The official code repo (in TensorFlow) can be found [here](https://github.com/krishnabits001/domain_specific_cl).

### Note on the naming convention
* `seg_models_v2` contains the latest UNet model. In case you want to run experiments, consider using this instead.
* `supervised_train.py` is for training the model for the Random-Random strategy shown in Table 1 of the paper. This should not be used for anything else.
* `loss.py` contains all the loss functions for pretraining. The function names are defined according to the convention chosen in the paper.
* `ssl_pretrain_encoder.py` contains the code for pretraining the UNet encoder with various strategies. The best model will be in the current working directory. 
    * The pretraining strategy is specified as one of the hyperparameters. Modify this parameter to change the pretraining strategy. 
* `finetune.py` should be used to load the pretrained encoder weights and finetune the model for the actual segmentation task.
* `scripts/` folder contains the bash scripts used for running fine-tuning experiments in a batch. A separate bash script is created for each pre-training strategy shown in the first half of Table 1 in the paper. The command-line inputs for running the experiments can be seen here.

### Installation
1. Clone this repository:
    ```py
    git clone https://github.com/naga-karthik/ssl-medical-imaging.git
    cd ssl-medical-imaging
    ```
2. Download the necessary packages:
    ```py
    pip install requirements.txt
    ```