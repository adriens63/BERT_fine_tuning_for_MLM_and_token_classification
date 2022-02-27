import torch
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import os
import os.path as osp
from datasets import load_metric
import json

from src.tools.model_summary import summary_parameters
from src.tools.timer import timeit
from src.tools.base_trainer import BaseTrainer






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
            model_name,
            weights_path,
            ) -> None:
        
        self.device = device
        self.model = model.to(device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.patience = patience
        self.train_data_loader = train_data_loader
        self.train_steps = train_steps
        self.val_data_loader = val_data_loader
        self.val_steps = val_steps
        self.checkpoint_frequency = checkpoint_frequency
        self.model_name = model_name
        self.weights_path = weights_path
        self.mod_dir = weights_path + model_name + '/'
        self.log_dir = weights_path + model_name + '/log_dir/'
        self.ckp_dir = weights_path + model_name + '/ckp_dir/'

        self.metric = load_metric('accuracy')
        self.tmp_metric = load_metric('accuracy')
        self.loss = {"train": [], "val": []}
        self.acc = {"train": [], "val": []}
        self.w = SummaryWriter(log_dir = self.log_dir)
        self.last_loss = np.inf
        self.trigger_times = 0



    def train(self) -> None:

        print('.... MLM CamemBERT training:')
        super().train()
        print('training done;')
        print()
        


    def _train_step(self) -> None:
        
        self.model.train()

        loop = tqdm(self.train_data_loader)
        running_loss = 0  
        n_batches = 0

        for i, batch in enumerate(loop):

            n_batches += 1
            self.optimizer.zero_grad()

            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            loss, out = self.model(input_ids, attention_mask = attention_mask, labels = labels)
            
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            predictions = torch.argmax(out, dim = -1)
            self.metric.add_batch(predictions = predictions.view(-1), references = batch["labels"].view(-1))
            self.tmp_metric.add_batch(predictions = predictions.view(-1), references = batch["labels"].view(-1))

            loop.set_postfix(loss = loss.item(), acc = self.tmp_metric.compute()['accuracy'])
            
            if i == self.train_steps:

                break
        
        self.loss['train'].append(running_loss / n_batches)
        acc = self.metric.compute()['accuracy']
        self.acc['train'].append(acc)



    def _val_step(self) -> None:

        self.model.eval()

        loop = tqdm(self.val_data_loader)
        running_loss = 0
        n_batches = 0

        for i, batch in enumerate(loop):

            n_batches += 1

            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            with torch.no_grad():

                loss, out = self.model(input_ids, attention_mask = attention_mask, labels = labels)

            running_loss += loss.item()

            predictions = torch.argmax(out, dim = -1)
            self.metric.add_batch(predictions = predictions.view(-1), references = batch["labels"].view(-1))
            self.tmp_metric.add_batch(predictions = predictions.view(-1), references = batch["labels"].view(-1))

            loop.set_postfix(loss = loss.item(), acc = self.tmp_metric.compute()['accuracy'])

            if i == self.val_steps:

                    break
            
        self.loss["val"].append(running_loss / n_batches)
        acc = self.metric.compute()['accuracy']
        self.acc["val"].append(acc)

