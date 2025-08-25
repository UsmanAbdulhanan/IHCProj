import os
import numpy as np

NUM_CLASSES = {
    'DISEASE_PROGR': 2,
    'OS_STATUS': 2,
    'cGS_PAT_1,cGS_PAT_2': 4,
}

PATCH_FEATS_DIM = {
    'resnet18': 512,
    'resnet50': 1024,
}

# Gleason grading constants
GG_TO_ONEHOT = {0: np.array([1, 0, 0, 0]),
                3: np.array([0, 1, 0, 0]),
                4: np.array([0, 0, 1, 0]),
                5: np.array([0, 0, 0, 1])}
GG_SUM_TO_LABEL = {0: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
INDEX_TO_GG = {0: 0, 1: 3, 2: 4, 3: 5}
ISUP_LABELS = {"0+0": 0, "3+3": 1, "3+4": 2, "4+3": 3, "4+4": 4,
               "3+5": 4, "5+3": 4, "4+5": 5, "5+4": 5, "5+5": 5}


def define_constants(
    base_path: str,
    is_real: bool,
    markers: str,
    feature_extractor: str,
    **kwargs
):
    # Markers
    markers_ = markers.split(',')
    GRAPHS_SAVE_DIR = {}
    FEATURES_SAVE_DIR = {}
    ADJ_SAVE_DIR = {}

    if is_real:
        for marker in markers_:
            GRAPHS_SAVE_DIR[marker] = os.path.join(base_path, 'patch_graphs', feature_extractor,  marker, 'real')
    else:
        for marker in markers_:
            if marker == 'HE':
                GRAPHS_SAVE_DIR[marker] = os.path.join(base_path, 'patch_graphs', feature_extractor, marker, 'real')
            else:
                GRAPHS_SAVE_DIR[marker] = os.path.join(base_path, 'patch_graphs', feature_extractor, marker, 'virtual')

    for marker, path in GRAPHS_SAVE_DIR.items():
        FEATURES_SAVE_DIR[marker] = os.path.join(path, 'feature_patches')
        ADJ_SAVE_DIR[marker] = os.path.join(path, 'adjacency_matrix')

    # Others
    METADATA_PATH = os.path.join(base_path, 'metadata.csv')
    DATA_SPLIT_DIR = os.path.join(base_path, 'data_splits', 'tasks')
    CHECKPOINTS_DIR = os.path.join(base_path, 'checkpoints', 'tasks')
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    return {
        'metadata_path': METADATA_PATH,
        'data_split_dir': DATA_SPLIT_DIR,
        'graphs_save_dir': GRAPHS_SAVE_DIR,
        'features_save_dir': FEATURES_SAVE_DIR,
        'adj_save_dir': ADJ_SAVE_DIR,
        'checkpoints_dir': CHECKPOINTS_DIR
    }
