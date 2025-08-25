import argparse
import yaml
import torch

from .train import trainer
from .test import Test
from .constant import define_constants
from i2iTranslation.utils.util import flatten, set_seed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # parameters
    parser.add_argument("--base_path",
                        type=str,
                        help='base directory of images, tissue masks, and model outcomes')
    parser.add_argument("--config_path",
                        type=str,
                        help='full path to config yaml file. sample configs are available in ./configs')
    parser.add_argument("--src_marker",
                        type=str,
                        help="name of the source marker",
                        default='HE')
    parser.add_argument("--dst_marker",
                        type=str,
                        help="name of the destination/target marker",
                        default='NKX3')
    parser.add_argument("--is_train",
                        type=eval,
                        default=False)
    parser.add_argument("--is_test",
                        type=eval,
                        default=False)
    args = parser.parse_args()
    args = vars(args)
    print('args: ', args)
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #### Train I2I: GAN loss + Multi-level consistency losses
   
        # define paths and constants
    constants = define_constants(**args)
    for key, val in constants.items():
        if key not in args.keys():
            args[key] = val

    # load i2i configs and update args
    with open(args['config_path']) as f:
        config = yaml.safe_load(f)
    args.update(config)
    args = flatten(args)
    if args['is_train']:
        # set device seed
        set_seed(device, args['train.params.seed'])

        # run trainer
        trainer(args, device)

    #### Test
    if args['is_test']:
        args['is_train'] = False
        print('testskin')
        # run tester
        test_obj = Test(args, device)
        test_obj.tester()
