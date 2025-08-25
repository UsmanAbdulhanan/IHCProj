import time
import numpy as np
from tqdm import tqdm
import torch
import logging
import os
from i2iTranslation.dataloader import ImagePatchDataset, TileDataset, prepare_image_patch_dataloader
from i2iTranslation.models import create_model
from i2iTranslation.models.global_objectives import convert_patches_to_image
from i2iTranslation.utils.util_bbox import load_bboxes, bbox_data_generator
from i2iTranslation.utils.util import delete_tensor_gpu



def setup_logger(log_dir, log_filename="training.log"):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()
logger = setup_logger("logs")

######################### sanity check for bboxes
# from PIL import Image
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# def plot_image(
#     image: np.ndarray
# ):
#     plt.imshow(image)
#     plt.show()
#     plt.close()

# def read_image(
#     image_fpath: str
# ) -> np.ndarray:
#     return np.array(Image.open(image_fpath))

# def plot_nuclei_labels(
#     image: np.ndarray, 
#     bbox_info: np.ndarray = None,
#     save_fpath: str = None
# ):
#     color_palette = {
#         0 : 'r',    # NEG nuclei
#         1 : 'b'     # POS nuclei
#     }
#     fig, ax = plt.subplots()
#     ax.imshow(Image.fromarray(image))
#     for k in range(len(bbox_info)):
#         a = bbox_info[k].tolist()
#         y0, x0, y1, x1, label = a
#         rect = patches.Rectangle((4*x0, 4*y0), 4*x1 - 4*x0, 4*y1 - 4*y0, linewidth=0.5, edgecolor=color_palette[label], facecolor='none')
#         ax.add_patch(rect)
    
#     if save_fpath is not None:
#         plt.savefig(save_fpath, dpi=600)

#     plot_image(image)
##############################################################################3333
def trainer(args, device):
    epoch_count = args['train.params.save.epoch_count']
    n_epochs = args['train.params.n_epochs']
    n_epochs_decay = args['train.params.n_epochs_decay']
    loss_logging_freq = args['train.params.save.loss_logging_freq']
    patch_batch_size = args['train.params.patch_batch_size']
    tile_batch_size = args['train.params.tile_batch_size']

    is_image_match = args['train.params.is_image_match']
    image_match_freq = args['train.params.image_match_freq']
    save_model_freq = args['train.params.save.save_model_freq']

    is_roi_discriminator =args['train.params.is_roi_discriminator']
    n_roi_bbox = args['train.params.n_roi_bbox']
    downsample = args['train.data.downsample']

    # get dataloader
    train_dataset = ImagePatchDataset(args)
    train_dataloader = prepare_image_patch_dataloader(args=args, dataset=train_dataset, shuffle=True)
    train_tile_dataset = TileDataset(args)
   
    # create model
    model = create_model(args, device)

    # Outer loop for epochs
    iter_count = 0
    for epoch in range(epoch_count, n_epochs + n_epochs_decay + 1):
        epoch_start_time = time.time()      # timer for entire epoch
        print('epoch: ', epoch)
        epoch_start_time = time.time()
        logger.info(f"Epoch {epoch} started.")

        # Inner loop within one epoch
        # Neighborhood objectives + Local objectives optimization
        for i, data in tqdm(enumerate(train_dataloader)):
            src_patches, dst_patches, src_patch_coords, dst_patch_coords, \
                src_img_size, dst_img_size, src_patch_paths, dst_patch_paths = data

            if not is_roi_discriminator:
                src_conditions, src_bboxes, dst_conditions, dst_bboxes = None, None, None, None
            else:
                # load bbox information
                src_bboxes = load_bboxes(src_patch_paths, downsample, args['src_bbox_info_path'])
                dst_bboxes = load_bboxes(dst_patch_paths, downsample, args['dst_bbox_info_path'])

                # remove patches without nuclei bboxes, then sample bboxes & create bbox conditions
                src_patches, src_conditions, src_bboxes, src_nobbox= bbox_data_generator(src_patches, src_bboxes, n_roi_bbox)
                dst_patches, dst_conditions, dst_bboxes, dst_nobbox= bbox_data_generator(dst_patches, dst_bboxes, n_roi_bbox)

                ################just a debug code to visualize the bbox################
                # plot_nuclei_labels(image=read_image(f'/home/comppec/Downloads/VirtualMultiplexer-master/data/i2i_patches/HE/{src_patch_paths[2]}'), bbox_info=src_bboxes[2], save_fpath=f"src_bbox_epoch_{epoch}_iter_{i}.png")
                # plot_nuclei_labels(image=read_image(f'/home/comppec/Downloads/VirtualMultiplexer-master/data/i2i_patches/NKX3/{dst_patch_paths[2]}'), bbox_info=dst_bboxes[2], save_fpath=f"dst_bbox_epoch_{epoch}_iter_{i}.png")
                #######################################

                validindices_src = torch.tensor([i for i in range(len(src_patches)) if i not in dst_nobbox])
                validindices_dst = torch.tensor([i for i in range(len(dst_patches)) if i not in src_nobbox])
                src_patches = src_patches[validindices_src]
                src_conditions = src_conditions[validindices_src]
                src_bboxes = [src_bboxes[k] for k in validindices_src]
                dst_patches = dst_patches[validindices_dst]
                dst_conditions = dst_conditions[validindices_dst]
                dst_bboxes = [dst_bboxes[k] for k in validindices_dst]
            
            ## Loop over src_patches and dst_patches in batches
            for j in range(0, min(len(src_patches), len(dst_patches)), patch_batch_size):
                src_patches_ = src_patches[j: j + patch_batch_size]
                dst_patches_ = dst_patches[j: j + patch_batch_size]

                # unpack data from dataset
                if not is_roi_discriminator:
                    model_input = {'src': src_patches_, 'dst': dst_patches_}
                else:
                    src_conditions_ = src_conditions[j: j + patch_batch_size]
                    src_bboxes_ = src_bboxes[j: j + patch_batch_size]
                    src_bboxes_ = [np.hstack((k * np.ones((bbox.shape[0], 1)), bbox)) for k, bbox in enumerate(src_bboxes_)]
                    src_bboxes_ = torch.Tensor(np.vstack(src_bboxes_))

                    dst_conditions_ = dst_conditions[j: j + patch_batch_size]
                    dst_bboxes_ = dst_bboxes[j: j + patch_batch_size]
                    dst_bboxes_ = [np.hstack((k * np.ones((bbox.shape[0], 1)), bbox)) for k, bbox in enumerate(dst_bboxes_)]
                    dst_bboxes_ = torch.Tensor(np.vstack(dst_bboxes_))

                    model_input = {'src': src_patches_, 'condition_src': src_conditions_, 'bboxes_src': src_bboxes_,
                                   'dst': dst_patches_, 'condition_dst': dst_conditions_, 'bboxes_dst': dst_bboxes_}

                # set model input data
                model.set_input(model_input)

                # data dependent initialization; loading pretrained weights; create schedulers
                if epoch == epoch_count and i == 0 and j == 0:
                    model.data_dependent_initialize(model_input)
                    model.setup(args)

                # calculate losses, gradients, update weights
                model.optimize_parameters()
                iter_count += 1

                # free up gpu memory
                delete_tensor_gpu(model_input)

                # loss logging at <loss_logging_freq>
                if iter_count % loss_logging_freq == 0:
                    losses = model.get_current_losses()
                    losses = {k: round(v, 4) for k, v in losses.items()}
                    logger.info(f"Iter {iter_count} {losses}")

        # Global objectives (Style & Content) optimization
        if is_image_match and epoch != 0 and epoch % image_match_freq == 0:
            for i, data in tqdm(enumerate(train_dataloader)):
                src_patches, dst_patches, src_patch_coords, dst_patch_coords, \
                    src_img_size, dst_img_size, src_patch_paths, dst_patch_paths = data

                # construct src, dst, and dst_fake images
                src_img, src_mask, dst_img, dst_mask, dst_img_fake = \
                    convert_patches_to_image(args, src_patches, src_img_size, src_patch_coords,
                                             dst_patches, dst_img_size, dst_patch_coords, model.netG, model.device)

                # preprocess the images before style and content matching
                src_tiles, dst_tiles, dst_fake_tiles = train_tile_dataset(
                    src_img, src_mask, dst_img, dst_mask, dst_img_fake
                )

                ## optimize style and content losses
                for j in range(0, len(src_tiles), tile_batch_size):
                    src_tiles_ = src_tiles[j: j + tile_batch_size]
                    dst_tiles_ = dst_tiles[j: j + tile_batch_size]
                    dst_fake_tiles_ = dst_fake_tiles[j: j + tile_batch_size]

                    # pass real and fake images
                    model.set_input_image({'src_real': src_tiles_, 'dst_real': dst_tiles_, 'dst_fake': dst_fake_tiles_})

                    # calculate style and content losses, gradients, update network weights
                    losses = model.optimize_parameters_image()
                    iter_count += 1

                    # free up gpu memory
                    delete_tensor_gpu({'src': src_tiles_, 'dst': dst_tiles_, 'dst_fake': dst_fake_tiles_})

                    # loss logging
                    if iter_count % loss_logging_freq == 0:
                        losses = {k: round(v, 4) for k, v in losses.items()}
                        print('val losses: ', losses)
                        logger.info(f"Iter, global loss {iter_count} {losses}")


        # update learning rates every epoch
        model.update_learning_rate()

        # validate the model every <val_epoch_freq> epochs
        if epoch != 0 and epoch % save_model_freq == 0:
            model.save_networks(epoch=epoch)
            logger.info(f"Model saved at epoch {epoch}")
    logger.info("Training complete.")