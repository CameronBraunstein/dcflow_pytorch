import argparse

def get_args(description):
    parser = argparse.ArgumentParser(description=description,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset',choices=['Sintel'],
                    help='dataset for training/ evaluation')
    parser.add_argument('--save_file',
                    help='name of path to save trained encoder')
    parser.add_argument('--training_downsample_factor',type=int, default=3,
                help='downsample factor for training images')
    parser.add_argument('--batch_size',type=int, default=30000,
                    help='size of batch of triples')
    parser.add_argument('--mini_batch_size',type=int, default=200,
                help='size of mini batch of triples. A mini batch will be processed all at once by the encoder during training')
    parser.add_argument('--patch_size',type=int, default=9,
                    help='width/height of a training triplet patch')
    parser.add_argument('--min_negative_offset',type=int, default=1,
                    help='minimum search for a negative patch in x or y direction')
    parser.add_argument('--max_negative_offset',type=int, default=5,
                    help='maximum search for a negative patch in x or y direction')
    return parser.parse_args()