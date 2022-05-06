import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
import os
import os.path as osp
from datasets import load_metric
import json

from src.tools.base_trainer import BaseTrainer






# ******************** constants *****************

NON_LBL_TOKEN = -100  # TODO: put these tokens in file
MAX_GRAD_NORM = 10
NUM_LABELS = 2






# ********************* trainer *********************

class Trainer(BaseTrainer):

    def __init__(
            self,
            device,
            model,
            epochs,
            batch_size,
            loss_fn,
            optimizer,
            lr_scheduler,
            patience,
            train_data_loader,
            train_steps,
            val_data_loader,
            val_steps,
            checkpoint_frequency,
            metric_frequency,
            model_name,
            weights_path
    ) -> None:

        super().__init__(
            device,
            model,
            epochs,
            batch_size,
            loss_fn,
            optimizer,
            lr_scheduler,
            patience,
            train_data_loader,
            train_steps,
            val_data_loader,
            val_steps,
            checkpoint_frequency,
            metric_frequency,
            model_name,
            weights_path,
        )



    def _train_step(self) -> None:

        self.model.train()

        loop = tqdm(self.train_data_loader)
        running_loss, running_accuracy = 0, 0  # loss and accuracy for the epoch
        n_batches = 0

        for i, batch in enumerate(loop):

            n_batches += 1
            self.optimizer.zero_grad()

            ids = batch['input_ids'].to(self.device)
            msk = batch['attention_mask'].to(self.device)
            lbl = batch['labels'].to(self.device)

            # out  = y_pred_logits
            loss, out = self.model(
                input_ids=ids, attention_mask=msk, labels=lbl).to_tuple()


            # loss
            epoch_loss = loss.sum().item()
            running_loss += epoch_loss

            # accuracy
            # from [b, seq_length] to [b * seq_length], we put all the lbl together to compare all at once
            flatten_lbl = lbl.view(-1)

            # from [b, seq_length, n_lbl] to [b * seq_length, n_lbl], we put all the predicted lbl together
            flatten_logits = out.view(-1, NUM_LABELS)

            # compute argmax along last axis to get shape [b * seq_lenght]
            flatten_pred = torch.argmax(flatten_logits, axis=1)


            # keeping only real lbls to perform comparison
            msk_unactive_lbl = flatten_lbl != NON_LBL_TOKEN

            flatten_real_lbl = torch.masked_select(
                flatten_lbl, mask=msk_unactive_lbl)
            flatten_real_pred = torch.masked_select(
                flatten_pred, mask=msk_unactive_lbl)

            batch_accuracy = (flatten_real_lbl ==
                              flatten_real_pred).sum() / self.batch_size
            running_accuracy += batch_accuracy

            # grad clipping
            torch.nn.utils.clip_grad_norm(
                parameters=self.model.parameters(), max_norm=MAX_GRAD_NORM)

            loss.sum().backward()
            self.optimizer.step()

            desc = {'loss': loss.item(), 'accuracy': batch_accuracy}
            loop.set_postfix(desc)

            if i == self.train_steps:

                break

        self.loss['train'].append(running_loss / n_batches)
        self.acc['train'].append(running_accuracy / n_batches)



    def _val_step(self) -> None:

        self.model.eval()

        loop = tqdm(self.val_data_loader)
        running_loss, running_accuracy = 0, 0
        n_batches = 0

        for i, batch in enumerate(loop):

            n_batches += 1

            ids = batch['input_ids'].to(self.device)
            msk = batch['attention_mask'].to(self.device)
            lbl = batch['labels'].to(self.device)

            with torch.no_grad():

                loss, out = self.model(ids, attention_mask=msk, labels=lbl).to_tuple()

            running_loss += loss.sum().item()

            # accuracy
            # from [b, seq_length] to [b * seq_length], we put all the lbl together to compare all at once
            flatten_lbl = lbl.view(-1)

            # from [b, seq_length, n_lbl] to [b * seq_length, n_lbl], we put all the predicted lbl together
            flatten_logits = out.view(-1, NUM_LABELS)

            # compute argmax along last axis to get shape [b, seq_lenght]
            flatten_pred = torch.argmax(flatten_logits, axis=1)

            # keeping only real lbls to perform comparison
            msk_unactive_lbl = flatten_lbl != NON_LBL_TOKEN

            flatten_real_lbl = torch.masked_select(
                flatten_lbl, mask=msk_unactive_lbl)
            flatten_real_pred = torch.masked_select(
                flatten_pred, mask=msk_unactive_lbl)

            batch_accuracy = (flatten_real_lbl ==
                              flatten_real_pred).sum() / self.batch_size
            running_accuracy += batch_accuracy


            if i == self.val_steps:

                break

        self.loss['val'].append(running_loss / n_batches)
        self.acc['val'].append(running_accuracy / n_batches)
