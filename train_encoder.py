import torch
import torch.nn as nn
from dataloader import get_dataloader
from patch_encoder import PatchEncoder
from spatial_correlation_sampler import SpatialCorrelationSampler,spatial_correlation_sample 

from multiprocessing import Process


def train(args):
    # Training is performed using SGD with momentum 0.9.
    momentum = 0.9
    # 10K iterations are performed with a learning rate of 10−1,
    # followed by 10K iterations with a learning rate of 10−2,
    # followed by 20K iterations with a learning rate of 10−3
    print(args.epoch_ranges[0])
    epoch_ranges = args.epoch_ranges
    learning_rates = args.learning_rates

    epoch_ranges = [5]
    learning_rates = [0.1]

    mini_batches_per_batch = args.batch_size//args.mini_batch_size

    loss_function = nn.TripletMarginLoss(p=2)
    dl = get_dataloader(args)
    dl.prepare_patch_mini_batch()
    pe = PatchEncoder(args)
    

    for epoch_range,learning_rate in zip(epoch_ranges,learning_rates):
        optimizer = torch.optim.SGD(pe.parameters(), lr=learning_rate, momentum=momentum)
        
        for epoch in range(epoch_range):
            print(epoch)
            bases, matches, negatives = dl.get_patch_mini_batch()
            p = Process(target=dl.prepare_patch_mini_batch, args=())
            p.start()


            anchor = pe.forward(torch.from_numpy(bases))[:,:,0,0]
            positive = pe.forward(torch.from_numpy(matches))[:,:,0,0]
            negative = pe.forward(torch.from_numpy(negatives))[:,:,0,0]

            print('test,', torch.dot(anchor[0],positive[0]))
            print('test,', torch.dot(anchor[1],negative[1]))
            print(anchor.shape,positive.shape)

            

            loss = loss_function(anchor, positive, negative)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            p.join()




# def compute_cost_volume(batch_1,batch_2):
#     cost_volume = torch.einsum('aijk,bijk->ajk', batch_1,batch_2)
#     return cost_volume