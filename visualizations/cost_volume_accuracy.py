from cost_volume_calculator import get_cost_volume_calculator
from dataloader import get_dataloader


def display_cost_volume_accuracy(args):
    dl = get_dataloader(args)
    cvc = get_cost_volume_calculator(args)
    frame_0,frame_1 = dl.get_frame_pair()
    cv = cvc.get_cost_volume(frame_0,frame_1)
    print(cv)

