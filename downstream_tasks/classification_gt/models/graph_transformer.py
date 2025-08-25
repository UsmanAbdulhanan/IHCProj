from typing import Dict, List
import copy
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import dense_mincut_pool

from vit import VisionTransformer, MLP
from gcn import GCNBlock
from weight_init import weight_init
from downstream_tasks.classification_gt.constant import NUM_CLASSES, PATCH_FEATS_DIM


class GraphTransformer(nn.Module):
    def __init__(
            self,
            args: Dict,
            **kwargs
    ):
        super(GraphTransformer, self).__init__()

        self.fusion_type = args['fusion_type']
        self.num_tasks = len(args['task_names'].split(','))
        self.num_markers = len(args['markers'].split(','))
        self.num_classes = NUM_CLASSES[args['task_names']]
        self.patch_feats_dim = PATCH_FEATS_DIM[args['feature_extractor']] \
            if 'patch_feats_dim' not in args.keys() else args['patch_feats_dim']
        self.embed_dim = args['embed_dim']
        self.num_node_cluster = args['num_node_cluster']
        self.mlp_ratio = args['mlp_ratio']

        # GNN module
        self.conv1 = GCNBlock(
            input_dim=self.patch_feats_dim,
            output_dim=self.embed_dim,
            **args
        )
        self.pool1 = nn.Linear(
            self.embed_dim,
            self.num_node_cluster
        )

        # Transformer
        self.transformer = VisionTransformer(**args)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        # MLP classification head/s
        self.heads = nn.ModuleList(
            [
                MLP(
                    in_features=self.embed_dim,
                    hidden_features=int(self.embed_dim * self.mlp_ratio),
                    out_features=self.num_classes
                ) for _ in range(self.num_tasks)
            ]
        )

        weight_init(self)

    def forward(
        self,
        node_feat: torch.Tensor,
        adj: torch.Tensor,
        mask: torch.Tensor
    ):
        # GNN
        X = mask.unsqueeze(2) * node_feat
        X = self.conv1(X, adj, mask)
        s = self.pool1(X)        

        # Projection: GNN to Transformer
        X, adj, mc_loss, ortho_loss = dense_mincut_pool(X, adj, s, mask)

        # Transformer
        b, _, _ = X.shape
        cls_token = self.cls_token.repeat(b, 1, 1)
        X = torch.cat([cls_token, X], dim=1)
        embedding = self.transformer(X)

        if self.fusion_type != 'late':
            x = []
            for head in self.heads:
                x.append(head(embedding))
            return x, mc_loss, ortho_loss
        else:
            return embedding, mc_loss, ortho_loss


class GraphTransformerLateFusion(nn.Module):
    def __init__(
            self,
            graph_transformers: nn.ModuleList,
            args: Dict,
            **kwargs
    ):
        super(GraphTransformerLateFusion, self).__init__()

        self.graph_transformers = graph_transformers
        self.num_markers = len(args['markers'].split(','))
        self.num_tasks = len(args['task_names'].split(','))
        self.num_classes = NUM_CLASSES[args['task_names']]
        self.embed_dim = args['embed_dim']
        self.aggregator_type = args['aggregator_type']
        self.aggregator_depth = args['aggregator_depth']
        self.mlp_ratio = args['mlp_ratio']

        # Transformer
        if self.aggregator_type == 'transformer':
            config = copy.deepcopy(args)
            config['depth'] = self.aggregator_depth
            self.aggregator = VisionTransformer(**config)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        elif self.aggregator_type == 'concat':
            self.aggregator = MLP(
                    in_features=self.embed_dim * self.num_markers,
                    hidden_features=int(self.embed_dim * self.mlp_ratio),
                    num_hidden_layers=self.aggregator_depth,
                    out_features=self.embed_dim
            )

        else:
            print('ERROR: invalid aggregator')
            exit()

        # MLP classification_gt head/s
        self.heads = nn.ModuleList([
            MLP(
                in_features=self.embed_dim,
                hidden_features=int(self.embed_dim * self.mlp_ratio),
                out_features=self.num_classes,
            ) for _ in range(self.num_tasks)
        ])

        weight_init(self)

    def forward(
            self,
            node_feat: List[torch.Tensor],
            adj: List[torch.Tensor],
            mask: List[torch.Tensor],
            missing_markers_index: torch.Tensor
    ):
        # marker wise forward pass
        embeddings = []
        mc_losses = []
        ortho_losses = []
        for i, (node_feat_, adj_, mask_, model_) in enumerate(zip(node_feat, adj, mask, self.graph_transformers)):
            embedding_, mc_loss_, ortho_loss_ = model_(node_feat_, adj_, mask_)

            # loss for frozen model
            if not self.is_finetune:
                mc_loss_ = torch.tensor(0).to(torch.float32).cuda()
                ortho_loss_ = torch.tensor(0).to(torch.float32).cuda()

            # zero-out missing markers
            embedding_ = embedding_ * missing_markers_index[:, i].unsqueeze(1)

            embeddings.append(embedding_)
            mc_losses.append(mc_loss_)
            ortho_losses.append(ortho_loss_)

        # Transformer-based aggregation
        if self.aggregator_type == 'transformer':
            embeddings = torch.stack(embeddings, dim=1)
            b, _, _ = embeddings.shape
            cls_token = self.cls_token.repeat(b, 1, 1)
            embeddings = torch.cat([cls_token, embeddings], dim=1)
            embeddings = self.aggregator(embeddings)

        # Concatenation
        elif self.aggregator_type == 'concat':
            embeddings = torch.cat(embeddings, dim=1)
            embeddings = self.aggregator(embeddings)

        else:
            print('ERROR: invalid aggregator')
            exit()

        # output task-specific heads
        x = []
        for head in self.heads:
            x.append(head(embeddings))
        return x, mc_losses, ortho_losses


class GraphTransformerEarlyFusion(nn.Module):
    def __init__(
            self,
            args: Dict,
            **kwargs
    ):
        super(GraphTransformerEarlyFusion, self).__init__()

        self.num_markers = len(args['markers'].split(','))
        self.num_tasks = len(args['task_names'].split(','))
        self.num_classes = NUM_CLASSES[args['task_names']]
        self.embed_dim = args['embed_dim']
        self.patch_feats_dim = PATCH_FEATS_DIM[args['feature_extractor']]

        # project node features into lower dim for parameter efficiency
        self.node_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features=self.patch_feats_dim, out_features=self.embed_dim, bias=False),
                nn.GELU()
            ) for _ in range(self.num_markers)
        ])

        # project concatenated (early-fused) node features
        self.fused_projector = nn.Linear(
            self.embed_dim * self.num_markers,
            self.embed_dim
        )

        # graph transformer - adjust patch embedding dim as per projected features dim
        config = copy.deepcopy(args)
        config['patch_feats_dim'] = self.embed_dim
        self.graph_transformer = GraphTransformer(args=config)

        weight_init(self)

    def forward(
            self,
            node_feat: List[torch.Tensor],
            adj: List[torch.Tensor],
            mask: List[torch.Tensor],
            missing_markers_index: torch.Tensor
    ):
        # project node features into lower dim
        node_feat_projected = []
        batch_size, num_nodes, _ = node_feat[0].shape

        for i, node_feat_ in enumerate(node_feat):
            node_feat_ = node_feat_.reshape(-1, self.patch_feats_dim)
            node_feat_ = self.node_projectors[i](node_feat_)
            node_feat_projected.append(node_feat_.reshape(batch_size, num_nodes, self.embed_dim))

        # process the data
        node_feats_ = torch.cat(node_feat_projected, dim=2)  # stack markers
        non_zero_adjs = []
        non_zero_masks = []
        for i, b in enumerate(missing_markers_index):
            # select only one non-zero marker. the rest have the same adj, mask
            non_zero_idx = np.nonzero(b.numpy())[0][0]
            non_zero_adjs.append(adj[non_zero_idx][i])
            non_zero_masks.append(mask[non_zero_idx][i])
        adjs_ = torch.stack(non_zero_adjs)
        masks_ = torch.stack(non_zero_masks)

        # project early fused node features
        node_feats_ = self.fused_projector(node_feats_)

        # graph transformer forward pass
        return self.graph_transformer(node_feats_, adjs_, masks_)
