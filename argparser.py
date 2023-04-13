import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Utility for training and evaluating DCFlow',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--action',choices=['Train', 'CostVolumeAccuracy'],
                    help='dataset for training/ evaluation')
    parser.add_argument('--cost_volume_algorithm',choices=['library_function', 'custom'],
                    help='how we all calculating cost volumes')
    parser.add_argument('--dataset',choices=['Sintel'],
                    help='dataset for training/ evaluation')
    parser.add_argument('--scene_name', choices=['alley_1'], default= 'alley_1',
                    help='specific Sintel scene')
    parser.add_argument('--training_or_test', choices=['training', 'test'], default='training',
                    help='training or test Sintel scene')
    parser.add_argument('--clean_or_final', choices=['clean','final'], default='final',
                    help='clean or final Sintel scene')
    parser.add_argument('--first_frame_number', type=int, default=1,
                    help='first frame number')
    
    parser.add_argument('--epoch_ranges', type=list,default=[10000, 10000, 20000],
                help='different epoch ranges with varying learning rates')
    parser.add_argument('--learning_rates', type=list,default=[0.1,0.01,0.001],
                help='different learning rates for each range of epoch')


    parser.add_argument('--save_file',
                    help='name of path to save trained encoder')
    parser.add_argument('--downsample_factor',type=int, default=3,
                help='downsample factor for dataloader')
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
    parser.add_argument('--feature_vector_length',type=int, default=64,
                    help='length of final encoded feature vector')
    return parser.parse_args()