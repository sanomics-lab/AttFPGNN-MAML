import os
import random
import joblib
import numpy as np
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.functional import auroc
from torch_geometric.data import DataLoader

from chem_lib.datasets import sample_meta_datasets, sample_test_datasets, MoleculeDataset
from chem_lib.utils import Logger
from maml_mol_relation_model import calculate_prototypes


class MamlMolRelationTrainer(nn.Module):

    def __init__(self, args, model) -> None:
        super(MamlMolRelationTrainer, self).__init__()

        self.args = args
        # self.model = MAML(model, lr=args.inner_lr, first_order=not args.second_order, anil=False, allow_unused=True)
        self.model = model
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.meta_lr, weight_decay=args.weight_decay)
        self.lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        
        self.criterion = nn.CrossEntropyLoss().to(args.device)

        self.dataset = args.dataset
        self.test_dataset = args.test_dataset if args.test_dataset is not None else args.dataset
        self.data_dir = args.data_dir
        self.train_tasks = args.train_tasks
        self.test_tasks = args.test_tasks
        self.n_shot_train = args.n_shot_train
        self.n_shot_test = args.n_shot_test
        self.n_query = args.n_query

        self.device = args.device

        self.emb_dim = args.emb_dim

        self.batch_task = args.batch_task

        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.inner_update_step = args.inner_update_step

        self.trial_path = args.trial_path
        trial_name = self.dataset + '_' + self.test_dataset + '@' + args.enc_gnn
        print(trial_name)
        logger = Logger(self.trial_path + '/results.txt', title=trial_name)
        log_names = ['Epoch']
        log_names += ['AUC-' + str(t) for t in args.test_tasks]
        log_names += ['AUC-Avg', 'AUC-Mid','AUC-Best']
        logger.set_names(log_names)
        self.logger = logger

        preload_train_data = {}
        if args.preload_train_data:
            print('preload train data')
            for task in self.train_tasks:
                dataset = MoleculeDataset(self.data_dir + self.dataset + "/new/" + str(task + 1),
                                          dataset=self.dataset)
                preload_train_data[task] = dataset
        preload_test_data = {}
        if args.preload_test_data:
            print('preload_test_data')
            for task in self.test_tasks:
                dataset = MoleculeDataset(self.data_dir + self.test_dataset + "/new/" + str(task + 1),
                                          dataset=self.test_dataset)
                preload_test_data[task] = dataset
        self.preload_train_data = preload_train_data
        self.preload_test_data = preload_test_data
        if 'train' in self.dataset and args.support_valid:
            val_data_name = self.dataset.replace('train','valid')
            print('preload_valid_data')
            preload_val_data = {}
            for task in self.train_tasks:
                dataset = MoleculeDataset(self.data_dir + val_data_name + "/new/" + str(task + 1),
                                          dataset=val_data_name)
                preload_val_data[task] = dataset
            self.preload_valid_data = preload_val_data

        self.train_epoch = 0
        self.best_auc = 0
        
        self.res_logs = []

    def loader_to_samples(self, data):
        loader = DataLoader(data, batch_size=len(data), shuffle=False, num_workers=0)
        for samples in loader:
            samples = samples.to(self.device)
            return samples

    def get_data_sample(self, task_id, train=True):
        if train:
            task = self.train_tasks[task_id]
            if task in self.preload_train_data:
                dataset = self.preload_train_data[task]
            else:
                dataset = MoleculeDataset(self.data_dir + self.dataset + "/new/" + str(task + 1), dataset=self.dataset)

            s_data, q_data = sample_meta_datasets(dataset, self.dataset, task,self.n_shot_train, self.n_query)

            s_data = self.loader_to_samples(s_data)
            q_data = self.loader_to_samples(q_data)

            adapt_data = {'s_data': s_data, 's_label': s_data.y, 'q_data': q_data, 'q_label': q_data.y,
                          'label': torch.cat([s_data.y, q_data.y], 0)}
            eval_data = { }
        else:
            task = self.test_tasks[task_id]
            if task in self.preload_test_data:
                dataset = self.preload_test_data[task]
            else:
                dataset = MoleculeDataset(self.data_dir + self.test_dataset + "/new/" + str(task + 1),
                                          dataset=self.test_dataset)
            s_data, q_data, q_data_adapt = sample_test_datasets(dataset, self.test_dataset, task, self.n_shot_test, self.n_query, self.update_step_test)
            s_data = self.loader_to_samples(s_data)
            q_loader = DataLoader(q_data, batch_size=self.n_query, shuffle=True, num_workers=0)
            q_loader_adapt = DataLoader(q_data_adapt, batch_size=self.n_query, shuffle=True, num_workers=0)

            adapt_data = {'s_data': s_data, 's_label': s_data.y, 'data_loader': q_loader_adapt}
            eval_data = {'s_data': s_data, 's_label': s_data.y, 'data_loader': q_loader}

        return adapt_data, eval_data

    def get_prediction(self, model, data, train=True):
        if train:
            logits, prototypes = model(data['s_data'], data['q_data'], data['s_label'])
            pred_dict = {'s_logits': s_logits, 'q_logits': q_logits, 'adj': adj, 'node_emb': node_emb}

        else:
            s_logits, logits,labels,adj_list,sup_labels = model.forward_query_loader(data['s_data'], data['data_loader'], data['s_label'])
            pred_dict = {'s_logits':s_logits, 'logits': logits, 'labels': labels,'adj':adj_list,'sup_labels':sup_labels}

        return pred_dict

    def get_adaptable_weights(self, model, adapt_weight=None):
        pass

    def get_loss(self, model, batch_data, pred_dict, train=True, flag = 0):
        pass

    def adapt_few_shot(self, adapt_data, inner_update_step):
        # TODO 修改 out_weights, out_bias 相关的, 之前忽略了 A P A^T 项
        support_feats, query_feats, support_labels = self.model(
            s_data=adapt_data['s_data'], q_data=adapt_data['q_data'], s_label=adapt_data['s_label']
        )
        # support_labels = adapt_data["s_label"]

        class_means, class_precisions = calculate_prototypes(support_feats, support_labels)

        local_model = deepcopy(self.model)
        local_model.train()
        local_optim = optim.SGD(local_model.parameters(), lr=self.args.inner_lr)
        local_optim.zero_grad()

        # Create output layer weights with prototype-based initialization
        init_weight = 2 * torch.matmul(class_means.unsqueeze(0).permute(1, 0, 2), class_precisions).squeeze(1)
        init_bias = -torch.mul(
            torch.matmul(class_means.unsqueeze(0).permute(1, 0, 2), class_precisions).squeeze(1),
            class_means
        ).sum(dim=1)

        output_weight = init_weight.detach().requires_grad_()
        output_bias = init_bias.detach().requires_grad_()

        # Optimize inner loop model on support set
        for _ in range(inner_update_step):
            # Determine loss on the support set
            support_feats, query_feats, support_labels = local_model(
                s_data=adapt_data['s_data'], q_data=adapt_data['q_data'], s_label=adapt_data['s_label']
            )
            support_preds = F.linear(support_feats, output_weight, output_bias)
            loss = F.cross_entropy(support_preds, support_labels.long())

            # Calculate gradients and perform inner loop update
            loss.backward()
            local_optim.step()
            # Update output layer via SGD
            output_weight.data -= self.args.inner_lr * output_weight.grad
            output_bias.data -= self.args.inner_lr * output_bias.grad
            # Reset gradients
            local_optim.zero_grad()
            output_weight.grad.fill_(0)
            output_bias.grad.fill_(0)

        # Re-attach computation graph of prototypes
        output_weight = (output_weight - init_weight).detach() + init_weight
        output_bias = (output_bias - init_bias).detach() + init_bias

        return local_model, output_weight, output_bias

    def train_step(self):
        self.train_epoch += 1

        task_id_list = list(range(len(self.train_tasks)))
        if self.batch_task > 0:
            batch_task = min(self.batch_task, len(task_id_list))
            task_id_list = random.sample(task_id_list, batch_task)
        data_batches = {}
        for task_id in task_id_list:
            db = self.get_data_sample(task_id, train=True)
            data_batches[task_id] = db

        for k in range(self.update_step):
            torch.set_grad_enabled(True)
            self.optimizer.zero_grad()

            # grad_accum = [0.0 for p in self.model.feature_extractor_params()]
            losses_eval = []

            for task_id in task_id_list:
                train_data, _ = data_batches[task_id]
                local_model, output_weight, output_bias = self.adapt_few_shot(
                    train_data, inner_update_step=self.inner_update_step
                )

                # Determine loss of query set
                support_feats, query_feats, support_labels = local_model(
                    s_data=train_data['s_data'], q_data=train_data['q_data'], s_label=train_data['s_label']
                )
                query_labels = train_data["q_label"]

                # local_model.compute_logits(support_feats, query_feats, support_labels)


                query_preds = F.linear(query_feats, output_weight, output_bias)
                loss = F.cross_entropy(query_preds, query_labels)
                acc = (query_preds.argmax(dim=1) == query_labels).float()

                loss.backward()
                for p_global, p_local in zip(self.model.parameters(), local_model.parameters()):
                    p_global.grad += p_local.grad  # First-order approx. -> add gradients of finetuned and base model

                losses_eval.append(loss.detach().cpu().numpy())

            if self.args.clip_value is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_value)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            losses_eval = np.mean(losses_eval)

            print('Train Epoch:', self.train_epoch, ', train update step:', k, ', loss_eval:', losses_eval)

        return self.model

    def test_step(self):
        step_results = {'query_preds': [], 'query_labels': [], 'task_index': []}
        auc_scores = []

        for task_id in range(len(self.test_tasks)):
            adapt_data, eval_data = self.get_data_sample(task_id, train=False)

            
            for i, batch in enumerate(adapt_data['data_loader']):
                batch = batch.to(self.device)
                cur_adapt_data = {'s_data': adapt_data['s_data'], 's_label': adapt_data['s_label'],
                                  'q_data': batch, 'q_label': None}

                local_model, output_weight, output_bias = self.adapt_few_shot(
                    cur_adapt_data, inner_update_step=self.update_step_test
                )

            local_model.eval()
            with torch.no_grad():
                query_feats, query_labels = local_model.forward_query_loader(
                    s_data=eval_data['s_data'], q_loader=eval_data['data_loader'], s_label=eval_data['s_label']
                )
                query_preds = F.linear(query_feats, output_weight, output_bias)
                query_preds = F.softmax(query_preds)
                auc = auroc(query_preds[:, 1], query_labels, task='binary').item()
                auc_scores.append(auc)

            print('Test Epoch:', self.train_epoch,', test for task:', task_id, ', AUC:', round(auc, 4))
            if self.args.save_logs:
                step_results['query_preds'].append(query_preds.cpu().numpy())
                step_results['query_labels'].append(query_labels.cpu().numpy())
                step_results['task_index'].append(self.test_tasks[task_id])

        mid_auc = np.median(auc_scores)
        avg_auc = np.mean(auc_scores)
        self.best_auc = max(self.best_auc, avg_auc)
        self.logger.append([self.train_epoch] + auc_scores + [avg_auc, mid_auc, self.best_auc], verbose=False)

        print('Test Epoch:', self.train_epoch, ', AUC_Mid:', round(mid_auc, 4), ', AUC_Avg: ', round(avg_auc, 4),
              ', Best_Avg_AUC: ', round(self.best_auc, 4),)
        
        if self.args.save_logs:
            self.res_logs.append(step_results)

        # self.model.load_state_dict(saved_state_dict)
        return self.best_auc

    def save_model(self):
        save_path = os.path.join(self.trial_path, f"step_{self.train_epoch}.pth")
        # torch.save(self.model.module.state_dict(), save_path)
        torch.save(self.model.state_dict(), save_path)
        print(f"Checkpoint saved in {save_path}")

    def save_result_log(self):
        joblib.dump(self.res_logs, self.args.trial_path+'/logs.pkl', compress=6)

    def conclude(self):
        df = self.logger.conclude()
        self.logger.close()
        print(df)
