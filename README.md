# ssl-medical-imaging

Link to the official code repo for reference: https://github.com/krishnabits001/domain_specific_cl

### Note on the naming convention
* `seg_models_v2` contains the latest UNet model. In case you want to run experiments, consider using this instead.
* `supervised_train.py` is for training the model for the Random-Random strategy shown in Table 1 of the paper. This should not be used for anything else.
* `loss.py` contains all the loss functions for pretraining. The function names are defined according to the convention chosen in the paper.
* `ssl_pretrain_encoder.py` contains the code for pretraining the UNet encoder with various strategies. The best model will be in the current working directory. 
    * The pretraining strategy is specified as one of the hyperparameters. Modigy this paramter to change the pretraining strategy. 
* `finetune.py` should be used to load the pretrained encoder weights and finetune the model for the actual segmentation task.

