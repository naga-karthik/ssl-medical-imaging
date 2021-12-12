# ssl-medical-imaging

Link to the official code repo for reference: https://github.com/krishnabits001/domain_specific_cl

### Note on the naming convention
* `seg_models_v2` contains the latest UNet model. In case you want to run experiments, consider using this file instead of the `seg_models` file.
* `supervised_train.py` is for training the model for the Random-Random strategy shown in Table 1 of the paper. This should not be used for anything else.
* `loss.py` contains all the loss functions for pretraining. The function names are defined according to the convention chosen in the paper.
* `ssl_pretrain_encoder.py` contains the code for pretraining the UNet encoder with the "GR" (simclr) strategy. The best model will be in the current working directory. Use `finetune.py` to load the pretrained encoder weights and finetune the model for the actual segmentation task.
* `ssl_pretrain_encoder_GDminus.py`, as  the name suggests, this file should be used for pretraining the encoder with the "GD-" strategy described in the paper. As mentioned above, use `finetune.py` for performing the downstream segmentation task.
