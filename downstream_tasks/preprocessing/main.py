import argparse
import os
import numpy as np
from PIL import Image
import torch
from glob import glob
from utils import extract_patch_features, get_torchvision_feature_extractor, get_transforms, extract_adj_matrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path",
                        type=str,
                        help='base directory of images, tissue masks, graphs, and model outcomes')
    parser.add_argument('--marker',
                        type=str,
                        help='name of the marker that needs to be encoded')
    parser.add_argument('--is_real',
                        type=eval,
                        help='flag indicating whether to process real/virtual images')
    parser.add_argument('--base_graphs_dir',
                        type=str,
                        help='directory path where the extracted graphs are to be saved')
    parser.add_argument('--feature_extractor',
                        type=str,
                        default='resnet50',
                        help='patch feature extractor')
    parser.add_argument('--patch_size',
                        type=int,
                        default=256)
    parser.add_argument('--stride',
                        type=int,
                        default=256)
    parser.add_argument('--patch_threshold',
                        type=float,
                        default=0.7)
    parser.add_argument('--batch_size',
                        type=int,
                        default=64)
    args = parser.parse_args()
    args = vars(args)

    # device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load feature extractor
    print('loading feature extractor ...')
    model = get_torchvision_feature_extractor(args["feature_extractor"])
    model = model.to(device)
    model.eval()

    # transforms
    transforms = get_transforms()

    # define constants
    if args['is_real']:
        base_img_dir = os.path.join(args['base_path'], 'images', args['marker'])
        base_tissue_mask_dir = os.path.join(args['base_path'], 'tissue_masks', args['marker'])
        base_graph_save_dir = os.path.join(args['base_graphs_dir'], 'patch_graphs', args['feature_extractor'], args['marker'], 'real')
    else:
        base_img_dir = os.path.join(args['base_path'], 'predictions', f'HE_{args["marker"]}', 'source_to_target')
        base_tissue_mask_dir = os.path.join(args['base_path'], 'HE', args['marker'])
        base_graph_save_dir = os.path.join(args['base_graphs_dir'], 'patch_graphs', args['feature_extractor'], args['marker'], 'virtual')

    features_save_dir = os.path.join(base_graph_save_dir, 'feature_patches')
    adj_save_dir = os.path.join(base_graph_save_dir, 'adjacency_matrix')
    os.makedirs(features_save_dir, exist_ok=True)
    os.makedirs(adj_save_dir, exist_ok=True)

    # process and extract patch-graph representation
    print('Processing marker: ', args['marker'])
    img_fpaths = glob(os.path.join(base_img_dir, '*.png'))
    print('#images to be processed: ', len(img_fpaths))

    for i, img_fpath in enumerate(img_fpaths):
        img_name = os.path.basename(img_fpath)[:-4]     # remove extension

        # read image
        image = np.asarray(Image.open(img_fpath))

        # read tissue mask
        tissue_mask = np.asarray(Image.open(os.path.join(base_tissue_mask_dir, f'{img_name}.png')))

        # extract patch feature
        features, coords = extract_patch_features(
            model=model,
            transforms=transforms,
            device=device,
            image=image,
            tissue_mask=tissue_mask,
            **args
        )
        torch.save(features, os.path.join(features_save_dir, f'{img_name}.pt'))

        # extract adjacency matrix
        adj_s = extract_adj_matrix(
            coords=coords,
            **args
        )
        torch.save(adj_s, os.path.join(adj_save_dir, f'{img_name}.pt'))
