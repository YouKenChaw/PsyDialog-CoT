import argparse

from dialogue.utils import print_args


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    return parser.parse_args()


def main(args):
    print_args(args)


if __name__ == '__main__':
    args = setup_args()
    main(args)
