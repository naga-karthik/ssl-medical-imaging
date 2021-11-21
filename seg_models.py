import torch 
import torch.nn as nn
from torch.nn.modules.activation import ReLU


def conv_bnorm_relu(in_feats, out_feats):
    return nn.Sequential(
        nn.Conv2d(in_feats, out_feats, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_feats),
        nn.ReLU(inplace=True)
    )

def upsample_conv_bnorm_relu(in_feats, out_feats):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(in_feats, out_feats, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_feats),
        nn.ReLU(inplace=True)
    )


class SegUnetEncoder_and_ProjectorG1(nn.Module):
    """
    Encoder of the UNet model and the Projector MLP on top of it
    """
    def __init__(self, in_channels=1, num_filters=[1, 16, 32, 64, 128, 128], fc_units=[3200, 1024], g1_out_dim=128):
        super(SegUnetEncoder_and_ProjectorG1, self).__init__()
        self.in_channels = in_channels
        self.num_filters = num_filters

        
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2)

        self.projector_g1 = nn.Sequential(
            nn.Linear(num_filters[5]*(3*3), fc_units[0]),    # back of a napkin math gives final enc6 output as 3x3 with base_n_filtersx32 filters
            nn.ReLU(inplace=True),
            nn.Linear(fc_units[0], fc_units[1]),
            nn.ReLU(inplace=True),
            nn.Linear(fc_units[1], g1_out_dim)
        )

        # Level 1 context pathway
        self.conv2d_c1_1 = conv_bnorm_relu(in_channels, num_filters[0])
        self.conv2d_c1_2 = conv_bnorm_relu(num_filters[0], num_filters[0])

        # Level 2 context pathway
        self.conv2d_c2_1 = conv_bnorm_relu(num_filters[0], num_filters[1])
        self.conv2d_c2_2 = conv_bnorm_relu(num_filters[1], num_filters[1])

        # Level 3 context pathway
        self.conv2d_c3_1 = conv_bnorm_relu(num_filters[1], num_filters[2])
        self.conv2d_c3_2 = conv_bnorm_relu(num_filters[2], num_filters[2])

        # Level 4 context pathway
        self.conv2d_c4_1 = conv_bnorm_relu(num_filters[2], num_filters[3])
        self.conv2d_c4_2 = conv_bnorm_relu(num_filters[3], num_filters[3])

        # Level 5 context pathway
        self.conv2d_c5_1 = conv_bnorm_relu(num_filters[3], num_filters[4])
        self.conv2d_c5_2 = conv_bnorm_relu(num_filters[4], num_filters[4])

        # Level 6 context pathway, level 0 localization pathway
        self.conv2d_c6_1 = conv_bnorm_relu(num_filters[4], num_filters[5])
        self.conv2d_c6_2 = conv_bnorm_relu(num_filters[5], num_filters[5])
        self.upsample_conv2d_block_l0 = upsample_conv_bnorm_relu(num_filters[5], num_filters[4])

    def forward(self, x):
        
        #  Level 1 context pathway
        out = self.conv2d_c1_1(x)
        residual_1 = out
        out = self.conv2d_c1_2(out)
        # Element Wise Summation
        out += residual_1
        out = self.maxpool2d(out)
        context_1 = out

        #  Level 2 context pathway
        out = self.conv2d_c2_1(out)
        residual_2 = out
        out = self.conv2d_c2_2(out)
        # Element Wise Summation
        out += residual_2
        out = self.maxpool2d(out)
        context_2 = out

        #  Level 3 context pathway
        out = self.conv2d_c3_1(out)
        residual_3 = out
        out = self.conv2d_c3_2(out)
        # Element Wise Summation
        out += residual_3
        out = self.maxpool2d(out)
        context_3 = out

        #  Level 4 context pathway
        out = self.conv2d_c4_1(out)
        residual_4 = out
        out = self.conv2d_c4_2(out)
        # Element Wise Summation
        out += residual_4
        out = self.maxpool2d(out)
        context_4 = out

        #  Level 5 context pathway
        out = self.conv2d_c5_1(out)
        residual_5 = out
        out = self.conv2d_c5_2(out)
        # Element Wise Summation
        out += residual_5
        out = self.maxpool2d(out)
        context_5 = out

        #  Level 6 context pathway, level 0 localization pathway
        out = self.conv2d_c6_1(out)
        residual_6 = out
        out = self.conv2d_c6_2(out)
        # Element Wise Summation
        out += residual_6
        out = self.maxpool2d(out)
        context_6 = out

        enc_out_c6 = out    # direct output from the encoder (use until here for encoder pretraining)
        # print(f"enc_out_c6 shape: {enc_out_c6.shape}")

        out_for_dec = self.upsample_conv2d_block_l0(enc_out_c6)     # pass this "out" onto the decoder path

        context_features = [context_1, context_2, context_3, context_4, context_5, context_6]   # for concatenating between the encoder and decoder (like in standard UNet)
        

        ### Defining the projector head ###
        out_projector_g1 = torch.flatten(enc_out_c6, 1)
        out_projector_g1 = self.projector_g1(out_projector_g1)


        return enc_out_c6, out_projector_g1,  out_for_dec, context_features



# testing with random inputs
if __name__ == "__main__":
    input_img = torch.randn(4, 1, 192, 192)    # batch_size x channels x H x W
    
    num_filters = [1, 16, 32, 64, 128, 128] 
    fc_units = [3200, 1024]
    
    model = SegUnetEncoder_and_ProjectorG1(in_channels=1, g1_out_dim=128, num_filters=num_filters, fc_units=fc_units)
    encoder_out, proj_out, out_dec, context_feats = model(input_img)
    print(f"encoder output shape: {encoder_out.shape}")
    print(f"projector output shape: {proj_out.shape}")
    print(f"output for decoder shape: {out_dec.shape}")
    print(f"context features shape: {[cf.shape for cf in context_feats]}")

