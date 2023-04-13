import torch
import numpy as np
from spatial_correlation_sampler import SpatialCorrelationSampler,spatial_correlation_sample 

# device = "cuda"
# batch_size = 1
# channel = 1
# H = 10
# W = 10
# dtype = torch.float32

# input1 = torch.randint(1, 4, (batch_size, channel, H, W), dtype=dtype, device=device, requires_grad=True)
# input2 = torch.randint_like(input1, 1, 4).requires_grad_(True)

# #You can either use the function or the module. Note that the module doesn't contain any parameter tensor.

# #function

# out = spatial_correlation_sample(input1,
# 	                         input2,
#                                  kernel_size=3,
#                                  patch_size=1,
#                                  stride=2,
#                                  padding=0,
#                                  dilation=2,
#                                  dilation_patch=1)

# #module

# correlation_sampler = SpatialCorrelationSampler(
#     kernel_size=3,
#     patch_size=1,
#     stride=2,
#     padding=0,
#     dilation=2,
#     dilation_patch=1)
# out = correlation_sampler(input1, input2)

class CostVolumeCalulator:
    def __init__(self,args):
        self.patch_size = args.patch_size
        return
    
class PretrainedCostVolumeCalculator(CostVolumeCalulator):
    def __init__(self,args):
        super(PretrainedCostVolumeCalculator,self).__init__(args)
        return
    
class LibraryCostVolumeCalculator(CostVolumeCalulator):
    def __init__(self,args):
        super(LibraryCostVolumeCalculator,self).__init__(args)
        return
    def get_cost_volume(self,frame_0,frame_1):
        height, width, channel = frame_0.shape

        input_shape = (1,channel,height,width)

        dtype = torch.float32
        device = "cuda"

        in_0 = torch.zeros(input_shape,dtype=dtype,device=device)
        in_0[0,:,:,:] = torch.from_numpy(np.transpose(frame_0,(2,0,1)))

        in_1 = torch.zeros(input_shape,dtype=dtype,device=device)
        in_1[0,:,:,:] = torch.from_numpy(np.transpose(frame_1,(2,0,1)))

  

        out = spatial_correlation_sample(in_0,in_1,
                                 kernel_size=3,
                                 patch_size=self.patch_size,
                                 stride=1,
                                 padding=0,
                                 dilation=2,
                                 dilation_patch=1)

        print(torch.max(out),torch.min(out))
        # H = 10
        # W = 10
        # 

        # input1 = torch.randint(1, 4, (batch_size, channel, H, W), dtype=dtype, device=device, requires_grad=True)
        # input2 = torch.randint_like(input1, 1, 4).requires_grad_(True)
        return 0


def get_cost_volume_calculator(args):
    if args.cost_volume_algorithm =='library_function':
        cvc = LibraryCostVolumeCalculator(args)
    else:
        raise Exception('No other calculators currently supported')
    return cvc
