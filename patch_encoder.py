
import torch.nn as nn

class PatchEncoder(nn.Module):
    def __init__(self,args):
        super(PatchEncoder,self).__init__()
        # Section 4: Our network has 4 convolutional layers.
        # Each of the first three layers uses 64 filters. Each convolu-
        # tion is followed by a pointwise truncation max(·, 0).
        # All filters are of size 3×3. We do not stride, pool, or pad.
        self.layer_1 = nn.Sequential()
        self.layer_1.add_module("Conv1", nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3))
        self.layer_1.add_module("ReLu1", nn.ReLU())

        self.layer_2 = nn.Sequential()
        self.layer_2.add_module("Conv2", nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3))
        self.layer_2.add_module("ReLu2", nn.ReLU())

        self.layer_3 = nn.Sequential()
        self.layer_3.add_module("Conv3", nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3))
        self.layer_3.add_module("ReLu3", nn.ReLU())


        # The last layer uses d filters and their output is normalized
        # to produce a unit-length feature vector f
        #Section 7: Unless stated
        # otherwise, a 64-dimensional feature embedding was used
        self.layer_4 = nn.Sequential()
        self.layer_4.add_module("Conv4", nn.Conv2d(in_channels=64, out_channels=args.feature_vector_length, kernel_size=3))
        #self.layer_4.add_module("Norm", nn.LayerNorm([args.feature_vector_length, 1, 1],elementwise_affine=False))
        #self.layer_4.add_module("Norm", )

        #self.conv1      = conv(self.batchNorm,   3,   64, kernel_size=7, stride=2)

    def forward(self,x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)

        x = nn.functional.normalize(x, p=2.0, dim = 1)
        return x
    