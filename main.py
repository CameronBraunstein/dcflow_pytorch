
from argparser import get_args
from train_encoder import train
from visualizations.cost_volume_accuracy import display_cost_volume_accuracy


if __name__ == '__main__':
    args = get_args()
    if args.action == 'Train':
        train(args)
    elif args.action == 'CostVolumeAccuracy':
        display_cost_volume_accuracy(args)
    else:
        raise Exception('No action specified for the utility')