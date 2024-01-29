import json
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple
from rdkit import Chem
import torch_scatter

import gpytorch

from .encoder import GNN_Encoder


all_fps = np.load("data/all_fps.npy")
all_pharm_graphs = np.load("data/all_pharm_graph.npy", allow_pickle=True)
with open("data/all_smis.list") as fr:
    all_smis = json.load(fr)
smi2id = {smi: idx for idx, smi in enumerate(all_smis)}

USE_ATTENTION = False
USE_MIXED_FPS = False
USE_PHARMACOPHORE = False

USE_XEMBEDDING = False

class MamlMolRelationModel(nn.Module):

    def __init__(self, args) -> None:
        super(MamlMolRelationModel, self).__init__()

        self.args = args
        self.gpu_id = args.gpu_id

        self.mol_encoder = GNN_Encoder(num_layer=args.enc_layer, emb_dim=args.emb_dim, JK=args.JK,
                                       drop_ratio=args.dropout, graph_pooling=args.enc_pooling, gnn_type=args.enc_gnn,
                                       batch_norm=args.enc_batch_norm)

        if args.pretrained:
            model_file = args.pretrained_weight_path
            if args.enc_gnn != 'gin':
                temp = model_file.split('/')
                model_file = '/'.join(temp[:-1]) + '/' + args.enc_gnn + '_' + temp[-1]
            print('load pretrained model from', model_file)
            self.mol_encoder.from_pretrained(model_file, self.gpu_id)

        self.emb_dim = self.args.emb_dim
        if USE_PHARMACOPHORE:
            self.pharmacophore_encoder = GNN_Encoder(
                num_layer=args.enc_layer, emb_dim=args.emb_dim, JK=args.JK,
                drop_ratio=args.dropout, graph_pooling=args.enc_pooling, gnn_type=args.enc_gnn,
                batch_norm=args.enc_batch_norm,
                use_xembedding=USE_XEMBEDDING
            )
            self.emb_dim += self.args.emb_dim

        if USE_MIXED_FPS:
            self.fp_fc = nn.Sequential(
                nn.Linear(self.emb_dim + 1489, 1024),
                nn.Dropout(p=0.3),
                nn.ReLU(),
                nn.Linear(1024, 512)
            )
        else:
            self.fp_fc = nn.Sequential(
                nn.Linear(self.emb_dim, 1024),
                nn.Dropout(p=0.3),
                nn.ReLU(),
                nn.Linear(1024, 512)
            )

        if USE_ATTENTION:
            padding_label_dim = 32
            self.multihead_attn = nn.MultiheadAttention(
                512 + padding_label_dim,
                num_heads=4,
                dropout=0
            )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def calculate_attented_feats(self, support_feats, query_feats, support_labels, padding_label_dim=32):
        support_features_flat, query_features_flat, support_labels = support_feats, query_feats, support_labels

        # 在分子之间做 attention 增强表示
        support_features_actives = support_features_flat[torch.where(support_labels == True)[0]]
        support_features_inactives = support_features_flat[torch.where(support_labels == False)[0]]

        if padding_label_dim > 0:
            query_features_flat = torch.cat(
                [query_features_flat, torch.zeros_like(query_features_flat[:, :padding_label_dim])], dim=1
            )
            support_features_actives = torch.cat(
                [support_features_actives, torch.ones_like(support_features_actives[:, :padding_label_dim])], dim=1
            )
            support_features_inactives = torch.cat(
                [support_features_inactives, (-1.) * torch.ones_like(support_features_inactives[:, :padding_label_dim])], dim=1
            )

        in_query = torch.cat([query_features_flat, support_features_actives, support_features_inactives], dim=0)
        in_key = torch.cat([support_features_actives, support_features_inactives], dim=0)
        in_value = torch.cat([support_features_actives, support_features_inactives], dim=0)

        updated_features = self.multihead_attn(query=in_query, key=in_key, value=in_value)[0]
        updated_features = updated_features + in_query

        query_features_updated, support_features_actives_updated, support_features_inactives_updated = torch.split(
            updated_features,
            split_size_or_sections=[query_features_flat.shape[0], support_features_actives.shape[0], support_features_inactives.shape[0]], 
            dim=0
        )

        if padding_label_dim > 0:
            query_features_updated = query_features_updated[:, :-padding_label_dim]
            support_features_actives_updated = support_features_actives_updated[:, :-padding_label_dim]
            support_features_inactives_updated = support_features_inactives_updated[:, :-padding_label_dim]

        support_features_updated = torch.cat([support_features_actives_updated, support_features_inactives_updated], dim=0)
        activate_size = support_features_actives_updated.shape[0]
        inactivate_size = support_features_inactives_updated.shape[0]
        support_labels = torch.tensor([True] * activate_size + [False] * inactivate_size).to(support_features_updated.device)

        return support_features_updated, query_features_updated, support_labels


    def forward_pharmacophore(self, batch_smis, graph_x=None):
        pharm_graphs = [all_pharm_graphs[smi2id[smi]] for smi in batch_smis]
        if graph_x is None:
            graph_x = np.concatenate([np.where(graph['x'] == 1)[1] for graph in pharm_graphs])[:, np.newaxis]
            graph_x = np.concatenate([graph_x, np.zeros_like(graph_x)], axis=1)
            graph_x = torch.tensor(graph_x).to(self.device)
        else:
            # TODO node_emb -> pharm_emb
            pool_idx = []
            pharm_cnt, atom_cnt = 0, 0

            for graph, smi in zip(pharm_graphs, batch_smis):
                pool_idx.append(graph["pool_index"] + np.array([[pharm_cnt, atom_cnt]] * graph["pool_index"].shape[0]))
                pharm_cnt += graph["x"].shape[0]
                atom_cnt += Chem.MolFromSmiles(smi).GetNumAtoms()
            pool_idx = np.concatenate(pool_idx)
            graph_x = torch_scatter.scatter(
                src=torch.tensor(graph_x[pool_idx[:, 1]]).to(self.device),
                index=torch.tensor(pool_idx[:, 0]).long().to(self.device),
                dim=0,
                reduce="sum"
            )

        graph_edge_index, graph_edge_attr, graph_batch = [], [], []
        for graph_idx, graph in enumerate(pharm_graphs):
            for edge_type_idx, edges in enumerate(graph["rg_edges"]):    
                graph_edge_index.append(edges + len(graph_batch))
                graph_edge_attr.extend([[edge_type_idx, 0]] * edges.shape[0])
            graph_batch.extend([graph_idx] * graph["x"].shape[0])
        graph_edge_index = np.concatenate(graph_edge_index)
        graph_edge_attr = np.array(graph_edge_attr)
        graph_batch = np.array(graph_batch)

        graph_edge_index = torch.tensor(graph_edge_index).T.int().to(self.device)
        graph_edge_attr = torch.tensor(graph_edge_attr).to(self.device)
        graph_batch = torch.tensor(graph_batch).to(self.device)

        graph_emb, group_emb = self.pharmacophore_encoder(graph_x, graph_edge_index, graph_edge_attr, graph_batch)
        return graph_emb, group_emb


    def forward(self, s_data, q_data, train_loss: bool=False, s_label=None, q_pred_adj=False, predictive_val_loss: bool=False, is_functional_call: bool=False):
        s_emb, s_node_emb = self.mol_encoder(s_data.x, s_data.edge_index, s_data.edge_attr, s_data.batch)
        q_emb, q_node_emb = self.mol_encoder(q_data.x, q_data.edge_index, q_data.edge_attr, q_data.batch)

        if USE_PHARMACOPHORE:
            if self.pharmacophore_encoder.use_xembedding:
                s_pharm_emb, _ = self.forward_pharmacophore(s_data.smiles)
                q_pharm_emb, _ = self.forward_pharmacophore(q_data.smiles)
            else:
                s_pharm_emb, _ = self.forward_pharmacophore(s_data.smiles, s_node_emb)
                q_pharm_emb, _ = self.forward_pharmacophore(q_data.smiles, q_node_emb)

            s_emb = torch.concat([s_emb, s_pharm_emb], dim=1)
            q_emb = torch.concat([q_emb, q_pharm_emb], dim=1)

        if USE_MIXED_FPS:
            s_fps = torch.tensor([all_fps[smi2id[smi]] for smi in s_data.smiles]).to(self.device)
            q_fps = torch.tensor([all_fps[smi2id[smi]] for smi in q_data.smiles]).to(self.device)

            s_features = self.fp_fc(torch.concat([s_emb, s_fps], dim=1))
            q_features = self.fp_fc(torch.concat([q_emb, q_fps], dim=1))
        else:
            s_features = self.fp_fc(s_emb)
            q_features = self.fp_fc(q_emb)

        if USE_ATTENTION:
            support_features_updated, query_features_updated, support_labels = self.calculate_attented_feats(
                s_features, q_features, s_label, padding_label_dim=32
            )
        else:
            # 分子表示不做任何进一步的更新
            support_features_updated = s_features
            query_features_updated = q_features
            support_labels = s_label

        return support_features_updated, query_features_updated, support_labels


    def forward_query_loader(self, s_data, q_loader, train_loss: bool=False, s_label=None, q_pred_adj=False, predictive_val_loss: bool=False, is_functional_call: bool=False):
        s_emb, s_node_emb = self.mol_encoder(s_data.x, s_data.edge_index, s_data.edge_attr, s_data.batch)

        if USE_PHARMACOPHORE:
            if self.pharmacophore_encoder.use_xembedding:
                s_pharm_emb, _ = self.forward_pharmacophore(s_data.smiles)
            else:
                s_pharm_emb, _ = self.forward_pharmacophore(s_data.smiles, s_node_emb)
            s_emb = torch.concat([s_emb, s_pharm_emb], dim=1)

        if USE_MIXED_FPS:
            s_fps = torch.tensor([all_fps[smi2id[smi]] for smi in s_data.smiles]).to(self.device)
            s_features = self.fp_fc(torch.concat([s_emb, s_fps], dim=1))
        else:
            s_features = self.fp_fc(s_emb)

        q_labels_list = []
        # q_logits_list = []
        q_features_list = []

        for q_data in q_loader:
            q_data = q_data.to(s_emb.device)
            q_labels_list.append(q_data.y)

            q_emb, q_node_emb = self.mol_encoder(q_data.x, q_data.edge_index, q_data.edge_attr, q_data.batch)

            if USE_PHARMACOPHORE:
                if self.pharmacophore_encoder.use_xembedding:
                    q_pharm_emb, _ = self.forward_pharmacophore(q_data.smiles)
                else:
                    q_pharm_emb, _ = self.forward_pharmacophore(q_data.smiles, q_node_emb)
                q_emb = torch.concat([q_emb, q_pharm_emb], dim=1)

            if USE_MIXED_FPS:
                q_fps = torch.tensor([all_fps[smi2id[smi]] for smi in q_data.smiles]).to(self.device)
                q_features = self.fp_fc(torch.concat([q_emb, q_fps], dim=1))
            else:
                q_features = self.fp_fc(q_emb)

            if USE_ATTENTION:
                support_features_updated, query_features_updated, support_labels = self.calculate_attented_feats(
                    s_features, q_features, s_label, padding_label_dim=32
                )
            else:
                # 分子表示不做任何进一步的更新
                support_features_updated = s_features
                query_features_updated = q_features
                support_labels = s_label

            # q_logit = self.compute_logits(support_features_updated, query_features_updated, support_labels)
            # q_logits_list.append(q_logit)
            q_features_list.append(query_features_updated)

        q_labels = torch.cat(q_labels_list, dim=0)
        # q_logits = torch.cat(q_logits_list, dim=0)
        q_features = torch.cat(q_features_list, dim=0)

        # q_preds = torch.softmax(q_logits, dim=1)
        return q_features, q_labels

    def compute_logits(self, support_features, query_features, support_labels):
        distance_metric = "mahalanobis"
        # distance_metric = "euclidean"
        if distance_metric == "mahalanobis":
            class_means, class_precision_matrices = calculate_prototypes(
                support_features,
                support_labels,
                distance_metric="mahalanobis"
            )
            # grabbing the number of classes and query examples for easier use later
            number_of_classes = class_means.size(0)
            number_of_targets = query_features.size(0)

            """
            Calculating the Mahalanobis distance between query examples and the class means
            including the class precision estimates in the calculations, reshaping the distances
            and multiplying by -1 to produce the sample logits
            """
            repeated_target = query_features.repeat(1, number_of_classes).view(
                -1, class_means.size(1)
            )
            repeated_class_means = class_means.repeat(number_of_targets, 1)
            repeated_difference = repeated_class_means - repeated_target
            repeated_difference = repeated_difference.view(
                number_of_targets, number_of_classes, repeated_difference.size(1)
            ).permute(1, 0, 2)
            first_half = torch.matmul(repeated_difference, class_precision_matrices)
            logits = torch.mul(first_half, repeated_difference).sum(dim=2).transpose(1, 0) * -1
            return logits, {"class_means": class_means, "class_precision": class_precision_matrices}
        else:  # euclidean
            class_prototypes, _ = calculate_prototypes(
                support_features,
                support_labels,
                distance_metric="euclidean"
            )
            logits = self._euclidean_distances(query_features, class_prototypes)
            return logits, {"class_means": class_means}

    @staticmethod
    def compute_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits, labels.long())

    def _euclidean_distances(
        self, query_features: torch.Tensor, class_prototypes: torch.Tensor
    ) -> torch.Tensor:
        num_query_features = query_features.shape[0]
        num_prototypes = class_prototypes.shape[0]

        distances = (
            (
                query_features.unsqueeze(1).expand(num_query_features, num_prototypes, -1)
                - class_prototypes.unsqueeze(0).expand(num_query_features, num_prototypes, -1)
            )
            .pow(2)
            .sum(dim=2)
        )

        return -distances


def _estimate_cov(
    examples: torch.Tensor, rowvar: bool = False, inplace: bool = False
) -> torch.Tensor:
    """
    SCM: Function based on the suggested implementation of Modar Tensai
    and his answer as noted in:
    https://discuss.pytorch.org/t/covariance-and-gradient-support/16217/5

    Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        examples: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    """
    if examples.dim() > 2:
        raise ValueError("m has more than 2 dimensions")
    if examples.dim() < 2:
        examples = examples.view(1, -1)
    if not rowvar and examples.size(0) != 1:
        examples = examples.t()
    factor = 1.0 / (examples.size(1) - 1)
    if inplace:
        examples -= torch.mean(examples, dim=1, keepdim=True)
    else:
        examples = examples - torch.mean(examples, dim=1, keepdim=True)
    examples_t = examples.t()
    return factor * examples.matmul(examples_t).squeeze()


def _extract_class_indices(labels: torch.Tensor, which_class: torch.Tensor) -> torch.Tensor:
    class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
    class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class
    return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector


def calculate_prototypes(support_feats, support_labels, distance_metric = "mahalanobis"):
    if distance_metric == "mahalanobis":
        means, precisions = [], []
        task_covariance_estimate = _estimate_cov(support_feats)
        for c in torch.unique(support_labels):
            # filter out feature vectors which have class c
            class_features = torch.index_select(support_feats, 0, _extract_class_indices(support_labels, c))
            # mean pooling examples to form class means
            means.append(torch.mean(class_features, dim=0, keepdim=True).squeeze())
            lambda_k_tau = class_features.size(0) / (class_features.size(0) + 1)
            lambda_k_tau = min(lambda_k_tau, 0.1)
            precisions.append(
                torch.inverse(
                    (lambda_k_tau * _estimate_cov(class_features))
                    + ((1 - lambda_k_tau) * task_covariance_estimate)
                    + 0.1
                    * torch.eye(class_features.size(1), class_features.size(1)).to(support_feats.device)
                )
            )

        means = torch.stack(means)
        precisions = torch.stack(precisions)
        return means, precisions
    else:
        means = []
        for c in torch.unique(support_labels):
            # filter out feature vectors which have class c
            class_features = torch.index_select(
                support_feats, 0, _extract_class_indices(support_labels, c)
            )
            means.append(torch.mean(class_features, dim=0))
        return torch.stack(means), None


if __name__ == "__main__":
    model = MamlMolRelationModel()
