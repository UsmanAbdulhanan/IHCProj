import argparse
import torch
import yaml

from constant import define_constants
from train_unimodal import Trainer as um_trainer
from train_mm_late import Trainer as mm_trainer_late
from train_mm_early import Trainer as mm_trainer_early


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path",
                        type=str,
                        help='base directory of images, tissue masks, graphs, and model outcomes')
    parser.add_argument("--config_path",
                        type=str,
                        help='full path to config yaml file. sample config is available in ./configs')
    parser.add_argument('--markers',
                        type=str,
                        default='HE,NKX3,AR,CD44,CD146,p53,ERG',
                        help='marker/s to use for analysis. names separated by ","')
    parser.add_argument('--is_real',
                        type=eval,
                        help='flag indicating whether to process real/virtual images')
    parser.add_argument('--task_names',
                        type=str,
                        help='task names separated by ",". useufl for multitask learning')
    parser.add_argument('--is_gleason',
                        type=str,
                        default=False,
                        help='flag indicating whether the task is gleason grading')
    args = parser.parse_args()
    args = vars(args)

    # define constants
    constants = define_constants(**args)
    args.update(constants)

    # load i2i configs and update args
    with open(args['i2i_config_path']) as f:
        config = yaml.safe_load(f)
    args.update(config)

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Train and Test
    num_markers = len(args['markers'].split(','))
    fusion_type = args['fusion_type']

    if num_markers == 1 and fusion_type == 'none':
        obj = um_trainer(args=args, device=device)

    elif num_markers > 1 and fusion_type == 'late':
        obj = mm_trainer_late(args=args, device=device)

    elif num_markers > 1 and fusion_type == 'early':
        obj = mm_trainer_early(args=args, device=device)

    if not args['is_test']:
        obj.train()
    obj.test()

