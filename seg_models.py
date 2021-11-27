import torch 
import torch.nn as nn
import torch.nn.functional as F
# from torch.nn.modules.activation import ReLU


def conv_bnorm_relu(in_feats, out_feats):
    return nn.Sequential(
        nn.Conv2d(in_feats, out_feats, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_feats),
        nn.ReLU()
    )

def upsample_conv_bnorm_relu(in_feats, out_feats):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(in_feats, out_feats, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_feats),
        nn.ReLU()
    )


class SegUnetEncoder_and_ProjectorG1(nn.Module):
    """
    Encoder of the UNet model and the Projector MLP on top of it
    """
    def __init__(self, in_channels=1, num_filters_list=[1, 16, 32, 64, 128, 128], fc_units_list=[3200, 1024], g1_out_dim=128):
        super(SegUnetEncoder_and_ProjectorG1, self).__init__()
        self.in_channels = in_channels
        self.num_filters = num_filters_list
        # self.context_features_list = []     # storing context features for concatenating between the encoder and decoder (like in standard UNet)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2)

        self.projector_g1 = nn.Sequential(
            nn.Linear(num_filters_list[5]*(6*6), fc_units_list[0]),    # final enc6 output is 6x6 (ie downsampled to 6x6), which is then multiplied by 128 filters for flattening
            nn.ReLU(),
            nn.Linear(fc_units_list[0], fc_units_list[1]),
            nn.ReLU(),
            nn.Linear(fc_units_list[1], g1_out_dim)
        )

        # Level 1 context pathway
        self.conv2d_c1_1 = conv_bnorm_relu(in_channels, num_filters_list[0])
        self.conv2d_c1_2 = conv_bnorm_relu(num_filters_list[0], num_filters_list[0])

        # Level 2 context pathway
        self.conv2d_c2_1 = conv_bnorm_relu(num_filters_list[0], num_filters_list[1])
        self.conv2d_c2_2 = conv_bnorm_relu(num_filters_list[1], num_filters_list[1])

        # Level 3 context pathway
        self.conv2d_c3_1 = conv_bnorm_relu(num_filters_list[1], num_filters_list[2])
        self.conv2d_c3_2 = conv_bnorm_relu(num_filters_list[2], num_filters_list[2])

        # Level 4 context pathway
        self.conv2d_c4_1 = conv_bnorm_relu(num_filters_list[2], num_filters_list[3])
        self.conv2d_c4_2 = conv_bnorm_relu(num_filters_list[3], num_filters_list[3])

        # Level 5 context pathway
        self.conv2d_c5_1 = conv_bnorm_relu(num_filters_list[3], num_filters_list[4])
        self.conv2d_c5_2 = conv_bnorm_relu(num_filters_list[4], num_filters_list[4])

        # Level 6 context pathway, level 0 localization pathway
        self.conv2d_c6_1 = conv_bnorm_relu(num_filters_list[4], num_filters_list[5])
        self.conv2d_c6_2 = conv_bnorm_relu(num_filters_list[5], num_filters_list[5])
        # self.upsample_conv2d_block_l0 = upsample_conv_bnorm_relu(num_filters_list[5], num_filters_list[4])

    def forward(self, x):
        
        context = []    # storing context features for concatenating between the encoder and decoder (like in standard UNet)
        #  Level 1 context pathway
        out = self.conv2d_c1_1(x)
        residual_1 = out
        out = self.conv2d_c1_2(out)
        # Element Wise Summation
        out += residual_1
        context.append(out)     # self.context_features_list.append(out)
        out = self.maxpool2d(out)

        #  Level 2 context pathway
        out = self.conv2d_c2_1(out)
        residual_2 = out
        out = self.conv2d_c2_2(out)
        # Element Wise Summation
        out += residual_2
        context.append(out)     # self.context_features_list.append(out)
        out = self.maxpool2d(out)

        #  Level 3 context pathway
        out = self.conv2d_c3_1(out)
        residual_3 = out
        out = self.conv2d_c3_2(out)
        # Element Wise Summation
        out += residual_3
        context.append(out)     # self.context_features_list.append(out)        
        out = self.maxpool2d(out)

        #  Level 4 context pathway
        out = self.conv2d_c4_1(out)
        residual_4 = out
        out = self.conv2d_c4_2(out)
        # Element Wise Summation
        out += residual_4
        context.append(out)     # self.context_features_list.append(out)
        out = self.maxpool2d(out)

        #  Level 5 context pathway
        out = self.conv2d_c5_1(out)
        residual_5 = out
        out = self.conv2d_c5_2(out)
        # Element Wise Summation
        out += residual_5
        context.append(out)     # self.context_features_list.append(out)        
        out = self.maxpool2d(out)

        #  Level 6 context pathway, level 0 localization pathway
        out = self.conv2d_c6_1(out)
        residual_6 = out
        out = self.conv2d_c6_2(out)
        # Element Wise Summation
        out += residual_6

        enc_out_c6 = out    # direct output from the encoder (use until here for encoder pretraining)

        ### Defining the projector head ###
        out_projector_g1 = torch.flatten(enc_out_c6, 1)
        out_projector_g1 = self.projector_g1(out_projector_g1)


        return enc_out_c6, out_projector_g1, context # self.context_features_list


class SegUnetDecoder(nn.Module):
    def __init__(self, num_classes=1, num_filters_list=[1, 16, 32, 64, 128, 128]):
        super(SegUnetDecoder, self).__init__()
        
        self.num_classes = num_classes
        self.num_filters = num_filters_list
        # self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.dec_c6_up = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec_dc6 = conv_bnorm_relu(num_filters_list[5], num_filters_list[4])
        self.dec_c5_1 = conv_bnorm_relu(2*num_filters_list[4], num_filters_list[4])
        self.dec_c5_2 = conv_bnorm_relu(num_filters_list[4], num_filters_list[4])

        self.dec_c5_up = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec_dc5 = conv_bnorm_relu(num_filters_list[4], num_filters_list[3])
        self.dec_c4_1 = conv_bnorm_relu(2*num_filters_list[3], num_filters_list[3])
        self.dec_c4_2 = conv_bnorm_relu(num_filters_list[3], num_filters_list[3])

        self.dec_c4_up = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec_dc4 = conv_bnorm_relu(num_filters_list[3], num_filters_list[2])
        self.dec_c3_1 = conv_bnorm_relu(2*num_filters_list[2], num_filters_list[2])
        self.dec_c3_2 = conv_bnorm_relu(num_filters_list[2], num_filters_list[2])

        self.dec_c3_up = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec_dc3 = conv_bnorm_relu(num_filters_list[2], num_filters_list[1])
        self.dec_c2_1 = conv_bnorm_relu(2*num_filters_list[1], num_filters_list[1])
        self.dec_c2_2 = conv_bnorm_relu(num_filters_list[1], num_filters_list[1])
    
        self.dec_c2_up = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec_dc2 = conv_bnorm_relu(num_filters_list[1], num_filters_list[0])
        self.dec_c1_1 = conv_bnorm_relu(2*num_filters_list[0], num_filters_list[0])

        self.conv2d_fin = nn.Conv2d(num_filters_list[0], num_classes, kernel_size=1, stride=1, padding=0, bias=False)

    
    def forward(self, x, context_features):
        # Get context features from the encoder
        context_1, context_2, context_3, context_4, context_5 = context_features

        # Level 5
        out = self.dec_c6_up(x)
        out = self.dec_dc6(out)
        out = torch.cat([out, context_5], dim=1)
        out = self.dec_c5_1(out)
        out = self.dec_c5_2(out)

        # Level 4
        out = self.dec_c5_up(out)
        out = self.dec_dc5(out)
        out = torch.cat([out, context_4], dim=1)
        out = self.dec_c4_1(out)
        out = self.dec_c4_2(out)

        # Level 3
        out = self.dec_c4_up(out)
        out = self.dec_dc4(out)
        out = torch.cat([out, context_3], dim=1)
        out = self.dec_c3_1(out)
        out = self.dec_c3_2(out)        

        # Level 2
        out = self.dec_c3_up(out)
        out = self.dec_dc3(out)
        out = torch.cat([out, context_2], dim=1)
        out = self.dec_c2_1(out)
        out = self.dec_c2_2(out)        

        # Level 1        
        out = self.dec_c2_up(out)
        out = self.dec_dc2(out)
        out = torch.cat([out, context_1], dim=1)
        out = self.dec_c1_1(out)

        # Unnormalized model output i.e. logits
        logits = self.conv2d_fin(out)

        # final activation layer
        out_final = F.softmax(logits, dim=1)

        return logits, out_final


class SegUnetFullModel(nn.Module):
    def __init__(self, in_channels=1, num_filters_list=[], fc_units=[], g1_out_dim=128, num_classes=1):
        super(SegUnetFullModel, self).__init__()
        self.unet_encoder = SegUnetEncoder_and_ProjectorG1(in_channels, num_filters_list, fc_units, g1_out_dim)
        self.unet_decoder = SegUnetDecoder(num_classes, num_filters_list)

    def forward(self, x):
        
        encoder_out, projector_out, context_features  = self.unet_encoder(x)
        logits, out_final = self.unet_decoder(encoder_out, context_features)

        return logits, out_final



# testing with random inputs
if __name__ == "__main__":
    input_img = torch.randn(8, 1, 192, 192)    # batch_size x channels x H x W
    
    in_channels=1
    num_filters = [1, 16, 32, 64, 128, 128] 
    fc_units = [3200, 1024]
    g1_out_dim=128
    num_classes=1
    
    encoder_model = SegUnetEncoder_and_ProjectorG1(in_channels=1, g1_out_dim=128, num_filters_list=num_filters, fc_units_list=fc_units)
    encoder_out, proj_out, context_feats = encoder_model(input_img)
    # print(f"encoder output shape: {encoder_out.shape}")
    # print(f"projector output shape: {proj_out.shape}")
    # print(f"context features shape: {[cf.shape for cf in context_feats]}")

    decoder_model = SegUnetDecoder(num_filters_list=num_filters)
    logits, out_final = decoder_model(encoder_out, context_feats)
    # print(f"logits shape: {logits.shape}")

    full_model = SegUnetFullModel(in_channels, num_filters, fc_units, g1_out_dim, num_classes)
    logits, out_final = full_model(input_img)
    print(f"full model logits shape: {logits.shape}")



