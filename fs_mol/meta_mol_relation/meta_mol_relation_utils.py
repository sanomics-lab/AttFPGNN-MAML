import logging
import os
import sys
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Tuple
from copy import deepcopy

import wandb
import numpy as np
import torch
import torch.nn.functional as F

from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))

from fs_mol.models.abstract_torch_fsmol_model import linear_warmup
from fs_mol.data import FSMolDataset, FSMolTaskSample, DataFold
from fs_mol.meta_mol_relation.meta_mol_relation_model import MetaMolRelationConfig, MetaMolRelationModel
from fs_mol.meta_mol_relation.meta_mol_relation_data import (
    get_metamolrelation_task_sample_iterable, 
    MetaMolRelationBatch,
    get_metamolrelation_batcher,
    task_sample_to_metamolrelation_task_sample
)
from fs_mol.models.abstract_torch_fsmol_model import MetricType
from fs_mol.utils.metrics import (
    BinaryEvalMetrics,
    compute_binary_task_metrics,
    avg_metrics_over_tasks,
    avg_task_metrics_list,
)
from fs_mol.utils.metric_logger import MetricLogger
from fs_mol.utils.torch_utils import torchify
from fs_mol.utils.test_utils import eval_model, FSMolTaskSampleEvalResults

from fs_mol.meta_mol_relation.cosine_annealing_warmup_scheduler import CosineAnnealingWarmupRestarts


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MetaMolRelationTrainerConfig(MetaMolRelationConfig):
    batch_size: int = 256
    tasks_per_batch: int = 16
    support_set_size: int = 16
    query_set_size: int = 256

    num_train_steps: int = 10000
    validate_every_num_steps: int = 50
    validation_support_set_sizes: Tuple[int] = (16, 128)
    validation_query_set_size: int = 256
    validation_num_samples: int = 5

    learning_rate: float = 0.001
    clip_value: Optional[float] = None
    use_attention: bool = False
    padding_label_dim: int = 0
    num_inner_steps: int = 1



def run_on_batches(
        model: MetaMolRelationModel,
        batches: List[MetaMolRelationBatch],
        batch_labels: List[torch.Tensor],
        train: bool = False,
        tasks_per_batch: int = 1,
        ) -> Tuple[float, BinaryEvalMetrics]:
    pass


def evaluate_metamolrelation_model(
        model: MetaMolRelationModel,
        dataset: FSMolDataset,
        support_sizes: List[int] = [16, 128],
        num_samples: int = 5,
        seed: int = 0,
        batch_size: int = 320,
        query_size: Optional[int] = None,
        data_fold: DataFold = DataFold.TEST,
        save_dir: Optional[str] = None,
    ) -> Dict[str, List[FSMolTaskSampleEvalResults]]:

    batcher = get_metamolrelation_batcher(max_num_graphs=batch_size)

    def test_model_fn(
        task_sample: FSMolTaskSample, temp_out_folder: str, seed: int
    ) -> BinaryEvalMetrics:
        # print(f"====== task name: {task_sample.name} ======")
        task_sample = torchify(
            task_sample_to_metamolrelation_task_sample(task_sample, batcher), device=model.device
        )

        task_preds, task_labels = [], []

        local_model, output_weight, output_bias = adapt_few_shot_mahalanobis(
            model, 
            batches=task_sample.batches, 
            batch_labels=task_sample.batch_labels
        )

        with torch.no_grad():
            for ibatch_features, ibatch_labels in zip(task_sample.batches, task_sample.batch_labels):
                torch.cuda.empty_cache()
                # support_logits, query_logits = local_model(task_sample.batches)

                # query_logits = query_logits.squeeze(-1)
                # query_labels = ibatch_features.query_labels
                # # query_loss = F.binary_cross_entropy_with_logits(query_logits, query_labels.float())
                # query_preds = torch.sigmoid(query_logits)

                support_feats, query_feats = local_model(ibatch_features)
                query_labels = ibatch_features.query_labels
                query_preds = F.linear(query_feats, output_weight, output_bias)
                # query_loss = F.cross_entropy(query_preds, query_labels.long())
                acc = (query_preds.argmax(dim=1) == query_labels.int()).float()

                # task_preds.append(query_preds.detach().cpu().numpy())
                task_preds.append(torch.softmax(query_preds, dim=-1)[:, 1].cpu().detach().numpy())
                task_labels.append(query_labels.detach().cpu().numpy())

            result_metrics = compute_binary_task_metrics(
                predictions=np.concatenate(task_preds, axis=0), 
                labels=np.concatenate(task_labels, axis=0)
            )

        # logger.info(
        #     f"{task_sample.task_name}:"
        #     f" {task_sample.num_support_samples:3d} support samples,"
        #     f" {task_sample.num_query_samples:3d} query samples."
        #     f" Avg. prec. {result_metrics.avg_precision:.5f}.",
        # )

        return result_metrics

    return eval_model(
        test_model_fn=test_model_fn,
        dataset=dataset,
        train_set_sample_sizes=support_sizes,
        out_dir=save_dir,
        num_samples=num_samples,
        # TODO 支持在验证集上选择最好的模型
        # valid_size_or_ratio=0.2,
        test_size_or_ratio=query_size,
        fold=data_fold,
        seed=seed,
    )


def validate_by_finetuning_on_tasks(
        model: MetaMolRelationModel,
        dataset: FSMolDataset,
        seed: int = 0,
        aml_run=None,
        metric_to_use: MetricType = "avg_precision",
    ) -> float:
    """
    Validation function for ADKTModel. Similar to test function;
    each validation task is used to evaluate the model more than once, the
    final results are a mean value for all tasks over the requested metric.
    """

    task_results = evaluate_metamolrelation_model(
        model,
        dataset,
        support_sizes=model.config.validation_support_set_sizes,
        num_samples=model.config.validation_num_samples,
        seed=seed,
        batch_size=model.config.batch_size,
        query_size=model.config.validation_query_set_size,
        data_fold=DataFold.VALIDATION,
    )

    # take the dictionary of task_results and return correct mean over all tasks
    mean_metrics = avg_metrics_over_tasks(task_results)
    if aml_run is not None:
        for metric_name, (metric_mean, _) in mean_metrics.items():
            aml_run.log(f"valid_task_test_{metric_name}", float(metric_mean))

    return mean_metrics[metric_to_use][0]



class MetaMolRelationModelTrainer(MetaMolRelationModel):

    def __init__(self, config: MetaMolRelationTrainerConfig):
        super().__init__(config)
        self.config = config
        # self.optimizer = torch.optim.Adam(self.parameters(), config.learning_rate)
        # self.lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        self.optimizer = torch.optim.AdamW(self.parameters(), config.learning_rate)
        self.lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = CosineAnnealingWarmupRestarts(
            self.optimizer,
            first_cycle_steps=2000,
            cycle_mult=1.0,
            max_lr=config.learning_rate,
            min_lr=config.learning_rate * 0.1,
            warmup_steps=100,
            gamma=1.0
        )


    def get_model_state(self) -> Dict[str, Any]:
        return {
            "model_config": self.config,
            "model_state_dict": self.state_dict(),
        }

    def save_model(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
    ):
        data = self.get_model_state()

        if optimizer is not None:
            data["optimizer_state_dict"] = optimizer.state_dict()
        if epoch is not None:
            data["epoch"] = epoch

        torch.save(data, path)

    def load_model_weights(
        self,
        path: str,
        load_task_specific_weights: bool,
        quiet: bool = False,
        device: Optional[torch.device] = None,
    ):
        pretrained_state_dict = torch.load(path, map_location=device)

        for name, param in pretrained_state_dict["model_state_dict"].items():
            if isinstance(param, torch.nn.Parameter):
                param = param.data
            self.state_dict()[name].copy_(param)

        optimizer_weights = pretrained_state_dict.get("optimizer_state_dict")
        if optimizer_weights is not None:
            for name, param in optimizer_weights.items():
                self.optimizer.state_dict()[name].copy_(param)

    def load_model_gnn_weights(
        self,
        path: str,
        device: Optional[torch.device] = None,
    ):
        pretrained_state_dict = torch.load(path, map_location=device)

        gnn_model_state_dict = pretrained_state_dict["model_state_dict"]
        our_state_dict = self.state_dict()

        # Load parameters (names specialised to GNNMultitask model), but also collect
        # parameters for GNN parts / rest, so that we can create a LR warmup schedule:
        gnn_params, other_params = [], []
        gnn_feature_extractor_param_name = "graph_feature_extractor."
        for our_name, our_param in our_state_dict.items():
            if (
                our_name.startswith(gnn_feature_extractor_param_name)
                and "final_norm_layer" not in our_name
            ):
                generic_name = our_name[len(gnn_feature_extractor_param_name) :]
                if generic_name.startswith("readout_layer."):
                    generic_name = f"readout{generic_name[len('readout_layer'):]}"
                our_param.copy_(gnn_model_state_dict[generic_name])
                logger.debug(f"I: Loaded parameter {our_name} from {generic_name} in {path}.")
                gnn_params.append(our_param)
            else:
                logger.debug(f"I: Not loading parameter {our_name}.")
                other_params.append(our_param)

        self.optimizer = torch.optim.Adam(
            [
                {"params": other_params, "lr": self.config.learning_rate},
                {"params": gnn_params, "lr": self.config.learning_rate / 10},
            ],
        )

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=self.optimizer,
            lr_lambda=[
                partial(linear_warmup, warmup_steps=0),  # for all params
                partial(linear_warmup, warmup_steps=100),  # for loaded GNN params
            ],
        )

    @classmethod
    def build_from_model_file(
        cls,
        model_file: str,
        config_overrides: Dict[str, Any] = {},
        quiet: bool = False,
        device: Optional[torch.device] = None,
    ) -> "MetaMolRelationModelTrainer":
        """Build the model architecture based on a saved checkpoint."""
        checkpoint = torch.load(model_file, map_location=device)
        config = checkpoint["model_config"]

        if not quiet:
            logger.info(f" Loading model configuration from {model_file}.")

        model = MetaMolRelationModelTrainer(config)
        model.load_model_weights(
            path=model_file,
            quiet=quiet,
            load_task_specific_weights=True,
            device=device,
        )
        return model
    
    def train_loop(self, out_dir: str, dataset: FSMolDataset, device: torch.device, aml_run=None):
        self.save_model(os.path.join(out_dir, "best_validation.pt"))

        train_task_sample_iterator = iter(
            get_metamolrelation_task_sample_iterable(
                dataset=dataset,
                data_fold=DataFold.TRAIN,
                num_samples=1,
                max_num_graphs=self.config.batch_size,
                support_size=self.config.support_set_size,
                query_size=self.config.query_set_size,
                repeat=True,
            )
        )

        best_validation_avg_prec = 0.0
        metric_logger = MetricLogger(
            log_fn=lambda msg: logger.info(msg),
            aml_run=aml_run,
            window_size=max(10, self.config.validate_every_num_steps / 5),
        )

        self.zero_grad()
        for step in range(1, self.config.num_train_steps + 1):
            torch.set_grad_enabled(True)
            # self.optimizer.zero_grad()

            task_batch_losses: List[float] = []
            task_batch_metrics: List[BinaryEvalMetrics] = []
            for _ in range(self.config.tasks_per_batch):
                task_sample = next(train_task_sample_iterator)
                train_task_sample = torchify(task_sample, device=device)

                # Perform inner loop adaptation
                local_model, output_weight, output_bias = adapt_few_shot_mahalanobis(
                    self,
                    batches=train_task_sample.batches,
                    batch_labels=train_task_sample.batch_labels
                )

                # calculate query set loss
                for ibatch_features, ibatch_lables in zip(train_task_sample.batches, train_task_sample.batch_labels):
                    # Perform inner loop adaptation

                    # local_model.zero_grad()
                    support_feats, query_feats = local_model(ibatch_features)
                    query_labels = ibatch_features.query_labels
                    query_preds = F.linear(query_feats, output_weight, output_bias)
                    query_loss = F.cross_entropy(query_preds, query_labels.long())
                    # acc = (query_preds.argmax(dim=1) == query_labels.int()).float()

                    task_metrics = compute_binary_task_metrics(
                        # predictions=query_preds.cpu().detach().numpy(), 
                        predictions=torch.softmax(query_preds, dim=-1)[:, 1].cpu().detach().numpy(),
                        labels=query_labels.cpu().detach().numpy()
                    )

                    task_batch_losses.append(query_loss.cpu().detach().numpy())
                    task_batch_metrics.append(task_metrics)

                    # Calculate gradients for query set loss
                    query_loss.backward(retain_graph=True)

                for p_global, p_local in zip(self.parameters(), local_model.parameters()):
                    # First-order approx. -> add gradients of finetuned and base model
                    if p_global.grad is None:
                        p_global.grad = p_local.grad
                    else:
                        p_global.grad += p_local.grad

            # TODO 根据 support set loss 得到新的参数 (attention + classifier)
            # TODO 根据 query set loss 做参数更新 (gnn + attention + classifier)

            # Now do a training step - run_on_batches will have accumulated gradients
            if self.config.clip_value is not None:
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.clip_value)
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            task_batch_mean_loss = np.mean(task_batch_losses)
            task_batch_avg_metrics = avg_task_metrics_list(task_batch_metrics)
            metric_logger.log_metrics(
                loss=task_batch_mean_loss,
                avg_prec=task_batch_avg_metrics["avg_precision"][0],
                kappa=task_batch_avg_metrics["kappa"][0],
                acc=task_batch_avg_metrics["acc"][0],
            )

            if step % self.config.validate_every_num_steps == 0:
                valid_metric = validate_by_finetuning_on_tasks(self, dataset, aml_run=aml_run)
                wandb.log({"valid avg prec": valid_metric})

                if aml_run:
                    # printing some measure of loss on all validation tasks.
                    aml_run.log(f"valid_mean_avg_prec", valid_metric)

                logger.info(
                    f"Validated at train step [{step}/{self.config.num_train_steps}],"
                    f" Valid Avg. Prec.: {valid_metric:.3f}",
                )

                # save model if validation avg prec is the best so far
                if valid_metric > best_validation_avg_prec:
                    best_validation_avg_prec = valid_metric
                    model_path = os.path.join(out_dir, "best_validation.pt")
                    self.save_model(model_path)
                    logger.info(f"Updated {model_path} to new best model at train step {step}")

        # save the fully trained model
        self.save_model(os.path.join(out_dir, "fully_trained.pt"))


def adapt_few_shot_mahalanobis(
        model: MetaMolRelationModel,
        batches: List[MetaMolRelationBatch],
        batch_labels: List[torch.Tensor]
    ):

    def _estimate_cov(
        examples: torch.Tensor, rowvar: bool = False, inplace: bool = False
    ) -> torch.Tensor:
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

    # TODO Determine prototype initialization
    support_feats, support_targets = [], []
    for ibatch_features, ibatch_labels in zip(batches, batch_labels):
        support_feats.append(model(ibatch_features)[0])
        support_targets.append(ibatch_features.support_labels)
    support_feats = torch.concat(support_feats, axis=0)
    support_targets = torch.concat(support_targets, axis=0)

    means = []
    precisions = []
    task_covariance_estimate = _estimate_cov(support_feats)
    for c in torch.unique(support_targets):
        # filter out feature vectors which have class c
        class_features = torch.index_select(support_feats, 0, _extract_class_indices(support_targets, c))
        # mean pooling examples to form class means
        means.append(torch.mean(class_features, dim=0, keepdim=True).squeeze())
        lambda_k_tau = class_features.size(0) / (class_features.size(0) + 1)
        lambda_k_tau = min(lambda_k_tau, 0.1)
        precisions.append(
            torch.inverse(
                (lambda_k_tau * _estimate_cov(class_features))
                + ((1 - lambda_k_tau) * task_covariance_estimate)
                + 0.1
                * torch.eye(class_features.size(1), class_features.size(1)).to(class_features.device)
            )
        )

    class_means = torch.stack(means)
    class_precision_matrices = torch.stack(precisions)

    # Create inner-loop model and optimizer
    local_model = deepcopy(model)
    local_model.train()
    lr_inner = 0.0001
    lr_output = 0.0001
    local_optim = torch.optim.SGD(local_model.parameters(), lr=lr_inner)
    local_optim.zero_grad()

    # TODO Create output layer weights with prototype-based initialization
    init_weight = 2 * torch.matmul(class_means.unsqueeze(0).permute(1, 0, 2), class_precision_matrices).squeeze(1)
    init_bias = -torch.mul(
        torch.matmul(class_means.unsqueeze(0).permute(1, 0, 2), class_precision_matrices).squeeze(1),
        class_means
    ).sum(dim=1)

    # init_weight = 2 * prototypes
    # init_bias = -torch.norm(prototypes, dim=1) ** 2
    output_weight = init_weight.detach().requires_grad_()
    output_bias = init_bias.detach().requires_grad_()

    # Optimize inner loop model on support set
    num_inner_steps = model.config.num_inner_steps
    for _ in range(num_inner_steps):
        for ibatch_features, ibatch_labels in zip(batches, batch_labels):
            # Determine loss on the support set
            # support_logits, query_logits = local_model(ibatch_features)
            # support_logits = support_logits.squeeze(-1)
            # support_labels = ibatch_features.support_labels
            # support_loss = F.binary_cross_entropy_with_logits(support_logits, support_labels.float())
            # support_preds = torch.sigmoid(support_logits)
            # support_preds = (support_preds > 0.5).int()
            # support_acc = (support_preds == support_labels).float()

            support_feats, query_feats = local_model(ibatch_features)
            support_labels = ibatch_features.support_labels
            support_preds = F.linear(support_feats, output_weight, output_bias)
            support_loss = F.cross_entropy(support_preds, support_labels.long())
            # acc = (preds.argmax(dim=1) == support_labels.int()).float()

            # Calculate gradients and perform inner loop update
            support_loss.backward()
            
            # TODO no local_optim
            local_optim.step()

            # TODO Update output layer via SGD & reset gradient
            # output_weight.data -= lr_output * output_weight.grad
            # output_bias.data -= lr_output * output_bias.grad

            with torch.no_grad():
                output_weight.copy_(output_weight - lr_output * output_weight.grad)
                output_bias.copy_(output_bias - lr_output * output_bias.grad)

            # Reset gradients
            local_optim.zero_grad()
            output_weight.grad.fill_(0)
            output_bias.grad.fill_(0)

    # TODO Re-attach computation graph of prototypes
    output_weight = (output_weight - init_weight).detach() + init_weight
    output_bias = (output_bias - init_bias).detach() + init_bias

    return local_model, output_weight, output_bias
