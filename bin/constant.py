import os

def define_constants(
    base_path: str,
    src_marker: str,
    dst_marker: str,
    **kwargs
):
    # Data split path
    DATA_SPLIT_PATH = os.path.join(base_path, 'data_splits', 'i2i')
    SRC_DATA_SPLIT_PATH = os.path.join(DATA_SPLIT_PATH, f'{src_marker}_splits.csv')
    DST_DATA_SPLIT_PATH = os.path.join(DATA_SPLIT_PATH, f'{dst_marker}_splits.csv')

    # Image paths
    IMAGES_PATH = os.path.join(base_path, 'images')
    SRC_IMAGES_PATH = os.path.join(IMAGES_PATH, src_marker)
    DST_IMAGES_PATH = os.path.join(IMAGES_PATH, dst_marker)

    # Tissue mask paths
    TISSUE_MASKS_PATH = os.path.join(base_path, 'tissue_masks')
    SRC_TISSUE_MASKS_PATH= os.path.join(TISSUE_MASKS_PATH, src_marker)
    DST_TISSUE_MASKS_PATH = os.path.join(TISSUE_MASKS_PATH, dst_marker)

    # I2I patch paths
    I2I_PATCHES_PATH = os.path.join(base_path, 'i2i_patches')
    SRC_I2I_PATCHES_PATH = os.path.join(I2I_PATCHES_PATH, src_marker)
    DST_I2I_PATCHES_PATH = os.path.join(I2I_PATCHES_PATH, dst_marker)

    # Path to bbox information
    SRC_BBOX_INFO_PATH = os.path.join(base_path, f'bbox_info/{src_marker}_{dst_marker}/{src_marker}_patch')
    DST_BBOX_INFO_PATH = os.path.join(base_path, f'bbox_info/{src_marker}_{dst_marker}/{dst_marker}_patch')

    # I2I checkpoints path
    I2I_CHECKPOINTS_PATH = os.path.join(base_path, 'checkpoints', 'i2i', f'{src_marker}_{dst_marker}')

    # Paths for I2I prediction (AB: forward prediction, BA: reverse prediction)
    I2I_PREDICTION_PATH = os.path.join(base_path, 'predictions', f'{src_marker}_{dst_marker}', 'source_to_target')

    return {
        'data_split_path': DATA_SPLIT_PATH,
        'src_data_split_path': SRC_DATA_SPLIT_PATH,
        'dst_data_split_path': DST_DATA_SPLIT_PATH,

        'images_path': IMAGES_PATH,
        'src_images_path': SRC_IMAGES_PATH,
        'dst_images_path': DST_IMAGES_PATH,

        'tissue_masks_path': TISSUE_MASKS_PATH,
        'src_tissue_masks_path': SRC_TISSUE_MASKS_PATH,
        'dst_tissue_masks_path': DST_TISSUE_MASKS_PATH,

        'i2i_patches_path': I2I_PATCHES_PATH,
        'src_i2i_patches_path': SRC_I2I_PATCHES_PATH,
        'dst_i2i_patches_path': DST_I2I_PATCHES_PATH,

        'i2i_checkpoints_path': I2I_CHECKPOINTS_PATH,
        'i2i_prediction_path': I2I_PREDICTION_PATH,

        'src_bbox_info_path': SRC_BBOX_INFO_PATH,
        'dst_bbox_info_path': DST_BBOX_INFO_PATH,
    }