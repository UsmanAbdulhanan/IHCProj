import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import copy

from dataloader import prepare_graph_dataloader, prepare_data_multimodal
from models.graph_transformer import GraphTransformer, GraphTransformerLateFusion
from models.losses import WeightedBCEWithLogitsLoss
from metrics import compute_metrics


class Trainer:
    def __init__(
            self,
            args,
            device
    ):
        self.args = args
        self.device = device
        self.is_test = args['is_test']

        # define model
        self.num_markers = len(args['markers'].split(','))
        models = []
        for i in range(self.num_markers):
            model = GraphTransformer(args=args)
            # remove 'heads' layer to get embeddings
            model.__delattr__('heads')
            models.append(model)

        self.model = GraphTransformerLateFusion(
            graph_transformers=nn.ModuleList(models),
            args=args
        )
        self.model = self.model.to(device)

        if not self.is_test:
            # prepare dataloaders and loss
            print('Preparing dataloaders ...')

            if args['weighted_sample']:
                self.train_dataloader, class_weights = prepare_graph_dataloader(args=args, mode='train',
                                                                                shuffle=False, weighted_sample=True)
                self.train_loss_criterion = WeightedBCEWithLogitsLoss(weights=class_weights)
            else:
                self.train_dataloader, class_weights = prepare_graph_dataloader(args=args, mode='train',
                                                                                shuffle=True, weighted_sample=False)
                self.train_loss_criterion = nn.BCEWithLogitsLoss()

            self.val_dataloader, _ = prepare_graph_dataloader(args=args, mode='val',
                                                              shuffle=False, weighted_sample=False)
            self.val_loss_criterion = nn.BCEWithLogitsLoss()

            # define optimizer, scheduler, and loss criteria
            self.optimizer = torch.optim.Adam(self.model.heads.parameters(), lr=args['lr'], weight_decay=5e-4)
            if args['scheduler'] == 'CosineAnnealing':
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=args['num_epochs']
                )
            elif args['scheduler'] == 'ExponentialLR':
                self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    self.optimizer, gamma=0.99
                )

        # define checkpoint dir
        self.checkpoints_dir = args['checkpoints_dir']

    def train(self):
        print('Start training...')
        best_val_weighted_f1 = 0

        for epoch in range(self.args['num_epochs']):
            print(f'Epoch ... {epoch}/{self.args["num_epochs"]}')
            self.model.train()

            losses = []
            logits = []
            targets = []
            for i, sample_batched in tqdm(enumerate(self.train_dataloader)):
                node_feats_, adjs_, masks_, missing_markers_index_, targets_ = \
                    prepare_data_multimodal(
                        batch_features=sample_batched['features'],
                        batch_adjs=sample_batched['adj'],
                        batch_targets=sample_batched['targets'],
                        device=self.device,
                        **self.args
                    )
                missing_markers_index_ = missing_markers_index_.to(self.device)

                self.optimizer.zero_grad()
                logits_, mc_loss_, ortho_loss_ = self.model(node_feats_, adjs_, masks_, missing_markers_index_)
                loss = torch.stack(mc_loss_).sum() + torch.stack(ortho_loss_).sum()
                for tgt, lgt in zip(targets_, logits_):
                    loss += self.train_loss_criterion(lgt, tgt.to(torch.float32))

                loss.backward()
                self.optimizer.step()

                losses.append(loss.detach().cpu().numpy())
                logits.append(logits_)
                targets.append(targets_)

            if self.scheduler is not None:
                self.scheduler.step()

            # metrics
            weighted_f1 = compute_metrics(targets, logits, **self.args)
            metrics = {
                'train_weighted_f1': weighted_f1,
                'train_loss': np.mean(losses)
            }
            print('train metrics: ', metrics)

            # validation
            if epoch > 0 and epoch % self.args['val_freq'] == 0:
                self.model.eval()

                losses = []
                logits = []
                targets = []
                with torch.no_grad():
                    for i, sample_batched in tqdm(enumerate(self.val_dataloader)):
                        node_feats_, adjs_, masks_, missing_markers_index_, targets_ = \
                            prepare_data_multimodal(
                                batch_features=sample_batched['features'],
                                batch_adjs=sample_batched['adj'],
                                batch_targets=sample_batched['targets'],
                                device=self.device,
                                **self.args
                            )
                        missing_markers_index_ = missing_markers_index_.to(self.device)

                        self.optimizer.zero_grad()
                        logits_, mc_loss_, ortho_loss_ = self.model(node_feats_, adjs_, masks_, missing_markers_index_)
                        loss = torch.stack(mc_loss_).sum() + torch.stack(ortho_loss_).sum()
                        for tgt, lgt in zip(targets_, logits_):
                            loss += self.val_loss_criterion(lgt, tgt.to(torch.float32))

                        losses.append(loss.detach().cpu().numpy())
                        logits.append(logits_)
                        targets.append(targets_)

                    # metrics
                    weighted_f1 = compute_metrics(targets, logits, **self.args)
                    val_loss = np.mean(losses)
                    metrics = {
                        'val_weighted_f1': weighted_f1,
                        'val_loss': val_loss
                    }
                    print('val metrics: ', metrics)

                # model selection: weighted F1
                if weighted_f1 > best_val_weighted_f1:
                    best_val_weighted_f1 = weighted_f1
                    torch.save(self.model.state_dict(), os.path.join(self.checkpoints_dir, 'best_val_weighted_f1_model.pth'))

    def test(self):
        # set testing on real data
        markers = self.args['markers'].split(',')
        features_save_dir = {}
        adj_save_dir = {}
        for marker in markers:
            graphs_save_dir = os.path.join(self.args['base_path'], 'patch_graphs', self.args['feature_extractor'], marker, 'real')
            features_save_dir[marker] = os.path.join(graphs_save_dir, 'feature_patches')
            adj_save_dir[marker] = os.path.join(graphs_save_dir, 'adjacency_matrix')
        self.args['features_save_dir'] = features_save_dir
        self.args['adj_save_dir'] = adj_save_dir

        # reset unnecessary parameters
        self.args['weighted_sample'] = False
        self.test_loss_criterion = nn.BCEWithLogitsLoss()

        # prepare dataloader
        self.test_dataloader, _ = prepare_graph_dataloader(args=self.args, mode='test',
                                                           shuffle=False, weighted_sample=False, drop_last=False)

        # load model state-dict
        model_path = os.path.join(self.checkpoints_dir, 'best_val_weighted_f1_model.pth')
        model = copy.deepcopy(self.model)
        model.load_state_dict(torch.load(model_path))
        model = model.to(self.device)
        model.eval()

        # run testing
        print(f'Start testing ...')
        logits = []
        targets = []
        with torch.no_grad():
            for sample_batched in tqdm(self.test_dataloader):
                node_feats_, adjs_, masks_, missing_markers_index_, targets_ = \
                    prepare_data_multimodal(
                        batch_features=sample_batched['features'],
                        batch_adjs=sample_batched['adj'],
                        batch_targets=sample_batched['targets'],
                        task_names=self.args['task_names'],
                        is_gleason=self.args['is_gleason'],
                        marker_dropout=0,   # no dropout during testing
                        device=self.device
                    )
                missing_markers_index_ = missing_markers_index_.to(self.device)

                logits_, _, _ = model(node_feats_, adjs_, masks_, missing_markers_index_)
                logits.append(logits_)
                targets.append(targets_)

            # metrics
            weighted_f1 = compute_metrics(
                targets=targets,
                logits=logits,
                **self.args
            )
            print('test weighted_f1: ', round(weighted_f1, 3))
