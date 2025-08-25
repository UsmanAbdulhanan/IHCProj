import copy
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from tqdm import tqdm
from constant import GG_TO_ONEHOT, NUM_CLASSES
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader, WeightedRandomSampler

def get_split(splits_path: str, split_key: str):
    all_splits = pd.read_csv(splits_path)
    split = all_splits[split_key]
    split = split.dropna().reset_index(drop=True)
    return split

def get_sample_ids(
    mode: str,
    data_split_dir: str,
    markers: List,
    task_names: str
):
    # unimodal, HE
    if len(markers) == 1:
        if 'HE' in markers:
            he_split = get_split(
                splits_path=os.path.join(data_split_dir, f'data_split_HE_valid_cores_{task_names}.csv'),
                split_key=f'{mode}_cores'
            )
            return he_split.values.tolist()
        else:
            union_split = get_split(
                splits_path=os.path.join(data_split_dir, f'data_split_union_HE_NKX3_AR_CD44_CD146_p53_{task_names}.csv'),
                split_key=f'{mode}_cores'
            )
            union_split = union_split.values.tolist()

            if mode != 'test':
                # intersection of the UNION split and the MARKER-specific split
                valid_cores = get_split(
                    splits_path=os.path.join(data_split_dir, f'data_split_{markers[0]}_valid_cores_{task_names}.csv'),
                    split_key=f'{mode}_cores'
                )
                valid_cores = valid_cores.values.tolist()
                return list(set(union_split) & set(valid_cores))
            else:
                return union_split
    else:
        union_split = get_split(
            splits_path=os.path.join(data_split_dir, f'data_split_union_HE_NKX3_AR_CD44_CD146_p53_{task_names}.csv'),
            split_key=f'{mode}_cores'
        )
        return union_split.values.tolist()


class GraphDataset(Dataset):
    """input and label image dataset"""

    def __init__(
            self,
            data_split_dir: str,
            markers: str,
            mode: str,
            features_save_dir: Dict,
            adj_save_dir: Dict,
            metadata_path: str,
            task_names: str,
            is_load_ram: bool=True,
            **kwargs
    ):
        super(GraphDataset, self).__init__()
        self.features_save_dir = features_save_dir
        self.adj_save_dir = adj_save_dir
        self.metadata_path = metadata_path
        self.task_names = task_names.split(',')
        self.markers = markers.split(',')
        self.is_load_ram = is_load_ram

        # get sample ids
        self.sample_ids = get_sample_ids(
            mode=mode,
            data_split_dir=data_split_dir,
            markers=self.markers,
            task_names=task_names
        )

        # self.sample_ids = self.sample_ids[:10]

        # get sample-wise metadata
        self.metadata = self._get_metadata()

        # pre-load features and adjacency matrices
        self.features = {}
        self.adj = {}
        if is_load_ram:
            self._load_in_ram()

        # get class weights
        self.task_wise_sample_weights, self.task_wise_weights_per_class = self._make_weights_for_balanced_classes()

    def _get_metadata(self):
        col_names = copy.deepcopy(self.task_names)
        col_names.append('PAT_ID')    # append patiend id for linking sample_ids to labels
        metadata = pd.read_csv(self.metadata_path, skipinitialspace=True, usecols=col_names)

        output = pd.DataFrame(columns=self.task_names)
        for sample_id in self.sample_ids:
            for i, row in metadata.iterrows():
                if str(row['PAT_ID']) in sample_id:
                    metadata_ = row
                    metadata_ = metadata_.drop('PAT_ID')
                    break
            output.loc[sample_id] = metadata_.astype(int)
        return output

    def _load_in_ram(self):
        print('Loading data into RAM ...')

        # load features and adjacency matrices
        for id in tqdm(self.sample_ids):
            self.features[id] = {}
            self.adj[id] = {}

            for marker in self.markers:
                # load features
                path = os.path.join(self.features_save_dir[marker], f'{id.replace("HE", marker)}.pt')
                if os.path.isfile(path):
                    self.features[id][marker] = torch.load(path)
                else:
                    self.features[id][marker] = None

                # load adjacency matrix
                path = os.path.join(self.adj_save_dir[marker], f'{id.replace("HE", marker)}.pt')
                if os.path.isfile(path):
                    self.adj[id][marker] = torch.load(path)
                else:
                    self.adj[id][marker] = None

    # to check for Gleason grading
    def _make_weights_for_balanced_classes(self, is_log=False):
        task_wise_sample_weights = {}
        task_wise_weights_per_class = {}

        for task_name in self.task_names:
            num_classes = NUM_CLASSES[task_name]

            count = np.zeros(num_classes)
            for id in self.sample_ids:
                label = self.metadata.loc[id][task_name]
                count[label] += 1
            print('count: ', count)

            # apply log transform
            if is_log:
                count = np.log(count)
                print('log count: ', count)

            N = float(np.sum(count))
            weight_per_class = N / count
            print('weight_per_class: ', weight_per_class ,'\n\n')

            weights = [0] * len(self.sample_ids)
            for i, id in enumerate(self.sample_ids):
                label = self.metadata.loc[id][task_name]
                weights[i] = weight_per_class[label]

            task_wise_sample_weights[task_name] = weights
            task_wise_weights_per_class[task_name] = weight_per_class

        return task_wise_sample_weights, task_wise_weights_per_class

    def __getitem__(self, index):
        id = self.sample_ids[index]
        features = []
        adj = []

        # get features
        if self.is_load_ram:
            for marker in self.markers:
                features.append(self.features[id][marker])
                adj.append(self.adj[id][marker])
        else:
            for marker in self.markers:
                # load features
                path = os.path.join(self.features_save_dir[marker], f'{id.replace("HE", marker)}.pt')
                features_ = torch.load(path) if os.path.isfile(path) else None
                features.append(features_)

                # load adjacency matrix
                path = os.path.join(self.adj_save_dir[marker], f'{id.replace("HE", marker)}.pt')
                adj_ = torch.load(path) if os.path.isfile(path) else None
                adj.append(adj_)

        targets = []
        for i in range(len(self.task_names)):
            targets.append(self.metadata.loc[id][self.task_names[i]])

        # prepare sample
        sample = {
            'id': id,
            'features': features,
            'adj': adj,
            'targets': targets
        }
        if features[0] is None:
            print(id)
        return sample

    def __len__(self):
        return len(self.sample_ids)


def collate(batch):
    id = [ b['id'] for b in batch ]
    features = [ b['features'] for b in batch ]         # B x [M x [N x D]]
    adj = [ b['adj'] for b in batch ]                   # B x [M x [N x N]]
    targets = [ b['targets'] for b in batch ]           # B x [T]

    return {
        'id': id,
        'features': features,
        'adj': adj,
        'targets': targets,
    }


def prepare_graph_dataloader(
        args,
        weighted_sample:bool,
        mode: str,
        shuffle: bool = False,
        drop_last: bool = True,
        **kwargs
) -> Tuple[DataLoader, np.ndarray]:

    dataset = GraphDataset(mode=mode, **args)

    # currently supporting only single-tasks
    if weighted_sample:
        task_name = args['task_names']
        sample_weights = dataset.task_wise_sample_weights[task_name]
        weights_per_class = dataset.task_wise_weights_per_class[task_name]
        dataloader = DataLoader(
            dataset,
            batch_size=args['batch_size'],
            num_workers=args['num_workers'],
            sampler=WeightedRandomSampler(sample_weights, len(sample_weights)),
            collate_fn=collate,
            shuffle=False,
            drop_last=drop_last
        )
    else:
        weights_per_class = None
        dataloader = DataLoader(
            dataset,
            batch_size=args['batch_size'],
            num_workers=args['num_workers'],
            collate_fn=collate,
            shuffle=shuffle,
            drop_last=drop_last
        )
    return dataloader, weights_per_class


def prepare_data_unimodal(
        batch_features, batch_adjs, batch_targets,
        task_names, is_gleason,
        device, **kwargs
):
    task_names = task_names.split(',')
    batch_size = len(batch_features)
    embed_dim = batch_features[0][0].shape[1]

    # prepare targets
    if len(task_names) == 1:
        batch_targets = [item for sublist in batch_targets for item in sublist]
        batch_targets = [F.one_hot(torch.Tensor(batch_targets).long(), num_classes=NUM_CLASSES[task_names[0]]).float()]
    else:
        targets = [[] for _ in range(len(task_names))]
        for b in batch_targets:
            for i, (task_name, tgt) in enumerate(zip(task_names, b)):
                if is_gleason:
                    targets[i].append(
                        torch.from_numpy(GG_TO_ONEHOT[tgt]).float()
                    )
                else:
                    targets[i].append(
                        F.one_hot(torch.Tensor(tgt).long(), num_classes=NUM_CLASSES[task_name]).float()
                    )
        batch_targets = [torch.stack(x) for x in targets]

    batch_targets = [targets.to(device) for targets in batch_targets]

    # flatten features and adjacency
    batch_features = [item for sublist in batch_features for item in sublist]
    batch_adjs = [item for sublist in batch_adjs for item in sublist]

    # batchify features and adjacency
    max_node_num = 0
    for i in range(batch_size):
        max_node_num = max(max_node_num, batch_features[i].shape[0])

    node_feats = torch.zeros(batch_size, max_node_num, embed_dim)
    adjs = torch.zeros(batch_size, max_node_num, max_node_num)
    masks = torch.zeros(batch_size, max_node_num)

    for i in range(batch_size):
        cur_node_num = batch_features[i].shape[0]

        # node features
        node_feats[i, 0:cur_node_num] = batch_features[i]

        # adjs
        adjs[i, 0:cur_node_num, 0:cur_node_num] = batch_adjs[i]

        # masks
        masks[i, 0:cur_node_num] = 1

    return node_feats.to(device), adjs.to(device), masks.to(device), batch_targets


def prepare_data_multimodal(
        batch_features, batch_adjs, batch_targets,
        task_names, is_gleason, marker_dropout,
        device, num_temp_nodes=5, drop_marker_idx=[], **kwargs
):
    batch_size = len(batch_features)
    num_markers = len(batch_features[0])
    embed_dim = batch_features[0][0].shape[1]
    task_names = task_names.split(',')
    is_drop_marker = True if len(drop_marker_idx) != 0 else False

    # prepare targets
    if len(task_names) == 1:
        batch_targets = [item for sublist in batch_targets for item in sublist]
        batch_targets = [F.one_hot(torch.Tensor(batch_targets).long(), num_classes=NUM_CLASSES[task_names[0]]).float()]
    else:
        targets = [[] for _ in range(len(task_names))]
        for b in batch_targets:
            for i, (task_name, tgt) in enumerate(zip(task_names, b)):
                if is_gleason:
                    targets[i].append(
                        torch.from_numpy(GG_TO_ONEHOT[tgt]).float()
                    )
                else:
                    targets[i].append(
                        F.one_hot(torch.Tensor(tgt).long(), num_classes=NUM_CLASSES[task_name]).float()
                    )
        batch_targets = [torch.stack(x) for x in targets]

    batch_targets = [targets.to(device) for targets in batch_targets]

    # prepare dropout matrix
    dropout_mat = torch.ones(size=(batch_size, num_markers))
    if marker_dropout > 0:
        for i in range(batch_size):
            # randomly select the number of markers to drop
            num_drop = np.random.randint(0, math.ceil(num_markers * marker_dropout))
            # randomly drop num_drop markers
            idx = np.random.choice(num_markers, num_drop, replace=False)
            dropout_mat[i, idx] = 0
        # ensure to always retain H&E
        dropout_mat[:, 0] = 1

    # dropping a pre-decided subset of markers
    if is_drop_marker:
        for drop_idx in drop_marker_idx:
            dropout_mat[:, drop_idx] = 0

    # prepare batch features, adjacency, mask
    node_feats = []
    adjs = []
    masks = []
    for t in range(num_markers):
        batch_features_marker = []
        batch_adjs_marker = []

        for i, b in enumerate(batch_features):
            if b[t] is not None:
                # enable marker dropout
                if marker_dropout > 0 or is_drop_marker:
                    batch_features_marker.append(
                        torch.zeros_like(b[t]) if dropout_mat[i, t] == 0 else b[t]
                    )
                else:
                    batch_features_marker.append(b[t])
            # missing marker scenario
            else:
                temp = torch.zeros(size=(num_temp_nodes, embed_dim))
                batch_features_marker.append(temp)
                dropout_mat[i, t] = 0

        for i, b in enumerate(batch_adjs):
            if b[t] is not None:
                # enable marker dropout
                if marker_dropout > 0 or is_drop_marker:
                    batch_adjs_marker.append(
                        torch.zeros_like(b[t]) if dropout_mat[i, t] == 0 else b[t]
                    )
                else:
                    batch_adjs_marker.append(b[t])
            # missing marker scenario
            else:
                temp = torch.zeros(size=(num_temp_nodes, num_temp_nodes))
                batch_adjs_marker.append(temp)

        # batchify features and adjacency
        max_node_num = 0
        for i in range(batch_size):
            max_node_num = max(max_node_num, batch_features_marker[i].shape[0])

        node_feats_marker = torch.zeros(batch_size, max_node_num, embed_dim)
        adjs_marker = torch.zeros(batch_size, max_node_num, max_node_num)
        masks_marker = torch.zeros(batch_size, max_node_num)
        for i in range(batch_size):
            cur_node_num = batch_features_marker[i].shape[0]

            # node features
            node_feats_marker[i, 0:cur_node_num] = batch_features_marker[i]

            # adjs
            adjs_marker[i, 0:cur_node_num, 0:cur_node_num] = batch_adjs_marker[i]

            # masks
            if dropout_mat[i, t] != 0:
                masks_marker[i, 0:cur_node_num] = 1

        node_feats.append(node_feats_marker.to(device))
        adjs.append(adjs_marker.to(device))
        masks.append(masks_marker.to(device))

    return node_feats, adjs, masks, dropout_mat, batch_targets


