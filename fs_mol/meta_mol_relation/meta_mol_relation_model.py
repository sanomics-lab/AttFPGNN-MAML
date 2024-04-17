from dataclasses import dataclass
from typing import List, Tuple
from typing_extensions import Literal

import torch
import torch.nn as nn

from fs_mol.modules.graph_feature_extractor import (
    GraphFeatureExtractor,
    GraphFeatureExtractorConfig,
)
from meta_mol_relation_data import MetaMolRelationBatch


FINGERPRINT_DIM = 2048
PHYS_CHEM_DESCRIPTORS_DIM = 42
MIXED_FP_DIM = 1489


@dataclass(frozen=True)
class MetaMolRelationConfig:
    # Model configuration:
    graph_feature_extractor_config: GraphFeatureExtractorConfig = GraphFeatureExtractorConfig()
    used_features: Literal[
        "gnn", "ecfp", "pc-descs", "gnn+ecfp", "ecfp+fc", "pc-descs+fc", "gnn+ecfp+pc-descs+fc",
        "gnn+ecfp+mixed+fc", "ecfp+mixed+fc"
    ] = "gnn+ecfp+fc"
    distance_metric: Literal["mahalanobis", "euclidean"] = "mahalanobis"


class MetaMolRelationModel(nn.Module):

    def __init__(self, config: MetaMolRelationConfig):
        super().__init__()
        self.config = config

        # Create GNN if needed:
        if self.config.used_features.startswith("gnn"):
            self.graph_feature_extractor = GraphFeatureExtractor(
                config.graph_feature_extractor_config
            )

        self.use_fc = self.config.used_features.endswith("+fc")
        self.normalizing_features = True

        # Create MLP if needed:
        # Determine dimension:
        fc_in_dim = 0
        if "gnn" in self.config.used_features:
            fc_in_dim += self.config.graph_feature_extractor_config.readout_config.output_dim
        if "ecfp" in self.config.used_features:
            fc_in_dim += FINGERPRINT_DIM
        if "pc-descs" in self.config.used_features:
            fc_in_dim += PHYS_CHEM_DESCRIPTORS_DIM
        if "mixed" in self.config.used_features:
            fc_in_dim += MIXED_FP_DIM

        self.padding_label_dim = self.config.padding_label_dim
        if self.use_fc:
            self.fc_out_dim = 512
            self.fc = nn.Sequential(
                nn.Linear(fc_in_dim, 1024),
                nn.ReLU(),
                nn.Linear(1024, self.fc_out_dim),
            )
            if self.config.use_attention:
                self.multihead_attn = nn.MultiheadAttention(self.fc_out_dim + self.padding_label_dim, num_heads=4, dropout=0.1)
        else:
            if self.config.use_attention:
                self.multihead_attn = nn.MultiheadAttention(fc_in_dim + self.padding_label_dim, num_heads=4, dropout=0.1)

        self.criterion = torch.nn.BCEWithLogitsLoss(reduction="none")


    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, input_batch: MetaMolRelationBatch, train_loss: bool=False):
        support_features: List[torch.Tensor] = []
        query_features: List[torch.Tensor] = []

        if "gnn" in self.config.used_features:
            support_features.append(self.graph_feature_extractor(input_batch.support_features))
            query_features.append(self.graph_feature_extractor(input_batch.query_features))
        if "ecfp" in self.config.used_features:
            support_features.append(input_batch.support_features.fingerprints)
            query_features.append(input_batch.query_features.fingerprints)
        if "pc-descs" in self.config.used_features:
            support_features.append(input_batch.support_features.descriptors)
            query_features.append(input_batch.query_features.descriptors)
        if "mixed" in self.config.used_features:
            support_features.append(input_batch.support_features.mixed_fps)
            query_features.append(input_batch.query_features.mixed_fps)


        support_features_flat = torch.cat(support_features, dim=1).float()
        query_features_flat = torch.cat(query_features, dim=1).float()

        if self.use_fc:
            support_features_flat = self.fc(support_features_flat)
            query_features_flat = self.fc(query_features_flat)

        if self.config.use_attention:
            """ start attention """
            if self.padding_label_dim > 0:
                query_features_flat = torch.cat(
                    [query_features_flat, torch.zeros_like(query_features_flat[:, :self.padding_label_dim])], dim=1
                )
                padding_emb_pos = torch.ones_like(support_features_flat[:, :self.padding_label_dim])
                padding_emb_neg = -1.0 * torch.ones_like(support_features_flat[:, :self.padding_label_dim])
                padding_emb = torch.where(
                    input_batch.support_labels.unsqueeze(1),
                    padding_emb_pos,
                    padding_emb_neg
                )
                support_features_flat = torch.cat([support_features_flat, padding_emb], dim=1)

            in_query = torch.cat([query_features_flat, support_features_flat], dim=0)
            updated_features = self.multihead_attn(
                query=in_query,
                key=support_features_flat,
                value=support_features_flat
            )[0]
            updated_features = updated_features + in_query

            query_features_updated, support_features_updated = torch.split(
                updated_features,
                split_size_or_sections=[query_features_flat.shape[0], support_features_flat.shape[0]],
                dim=0
            )

            if self.padding_label_dim > 0:
                query_features_updated = query_features_updated[:, :-self.padding_label_dim]
                support_features_updated = support_features_updated[:, :-self.padding_label_dim]

            if self.normalizing_features:
                support_features_updated = torch.nn.functional.normalize(support_features_updated, p=2, dim=1)
                query_features_updated = torch.nn.functional.normalize(query_features_updated, p=2, dim=1)
            """ end attention """
        else:
            if self.normalizing_features:
                support_features_flat = torch.nn.functional.normalize(support_features_flat, p=2, dim=1)
                query_features_flat = torch.nn.functional.normalize(query_features_flat, p=2, dim=1)

            support_features_updated = support_features_flat
            query_features_updated = query_features_flat

        return support_features_updated, query_features_updated
