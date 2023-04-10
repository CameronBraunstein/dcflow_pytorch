from argparser import get_args
from dataloader import SintelDataloader

def train(args):
    dl = SintelDataloader(args)
    dl.prepare_patch_mini_batch()
    print('HELLO')



if __name__ == '__main__':
    args = get_args('Train the feature encoder')
    train(args)