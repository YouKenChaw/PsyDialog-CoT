import argparse

from accelerate import Accelerator
from torch.utils.tensorborad import SummaryWriter

from dialogue.utils import print_args


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    return parser.parse_args()


def main(args):
    accelerator = Accelerator()

    if accelerator.is_main_process:
    print_args(args)


if __name__ == '__main__':
    args = setup_args()
    main(args)
