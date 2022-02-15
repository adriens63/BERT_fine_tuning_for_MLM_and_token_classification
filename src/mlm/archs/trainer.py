import torch
from tqdm import tqdm
import os
import os.path as osp
from datasets import load_metric
import numpy as np
import json




# ********************* trainer *********************

class Trainer:

    def __init__(
            self,
            device,
            model,
            epochs,
            batch_size,
            loss_fn,
            optimizer,
            lr_scheduler,
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
        self.train_data_loader = train_data_loader
        self.train_steps = train_steps
        self.val_data_loader = val_data_loader
        self.val_steps = val_steps
        self.checkpoint_frequency = checkpoint_frequency
        self.model_name = model_name
        self.weights_path = weights_path
        self.mod_dir = weights_path + model_name + '/'
        #self.log_dir = weights_path + model_name + '/log_dir/'
        self.ckp_dir = weights_path + model_name + '/ckp_dir/'

        self.metric = load_metric('accuracy')
        self.loss = {"train": [], "val": []}


    def train(self) -> None:

        print('.... Start training')

        for e in range(self.epochs):
            self._train_step()
            self._val_step()
            print(
                "Epoch: {}/{}, Train Loss={:.5f}, Val Loss={:.5f}".format(
                    e + 1,
                    self.epochs,
                    self.loss["train"][-1],
                    self.loss["val"][-1],
                )
            )
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if self.checkpoint_frequency:
                    self._save_checkpoint(e)


        print('done;')
        print()

    def _train_step(self) -> None:
        
        self.model.train()

        loop = tqdm(self.train_data_loader)
        running_loss = []  

        for i, batch in enumerate(loop):
            self.optimizer.zero_grad()

            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            out = self.model(input_ids, attention_mask = attention_mask, labels = labels)
            
            #TODO mettre self.loss_fn ici
            #TODO enlever les com
            loss = out.loss
            loss.backward()
            self.optimizer.step()

            running_loss.append(loss.item())

            loop.set_postfix(loss = loss.item())
            
            if i == self.train_steps:
                break
        
        epoch_loss = np.mean(running_loss) #TODO: on utilise np ou pas
        self.loss["train"].append(epoch_loss)
        


    def _val_step(self) -> None:

        self.model.eval()

        loop = tqdm(self.val_data_loader)
        running_loss = []

        for i, batch in enumerate(loop):

            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            with torch.no_grad():

                out = self.model(input_ids, attention_mask = attention_mask, labels = labels)

            loss = out.loss

            running_loss.append(loss.item())

            logits = out.logits
            predictions = torch.argmax(logits, dim= -1)
            self.metric.add_batch(predictions=predictions, references=batch["labels"])

            if i == self.val_steps:
                    break

        epoch_loss = np.mean(running_loss)
        self.loss["val"].append(epoch_loss)
        self.metric.compute()


    def _save_checkpoint(self, epoch: int) -> None:
        """Save model checkpoint to `self.model_dir` directory"""

        epoch_num = epoch + 1
        if epoch_num % self.checkpoint_frequency == 0:
            
            print('.... Saving ckp')
            model_path = "checkpoint_{}.pt".format(str(epoch_num).zfill(3))
            model_path = osp.join(self.ckp_dir, model_path)
            torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'epoch': epoch
                    }, model_path)
            print('done;')
            print()


    def save_model(self) -> None:

        print('.... Save model')
        
        model_path = osp.join(self.weights_path, self.model_name)
        
        if not osp.exists(model_path):
            os.makedirs(model_path)

        
        torch.save(self.model, model_path + '/' + self.model_name + '.pt')

        print('done;')
        print()

    
    def save_loss(self) -> None:

        """Save train/val loss as json file to `self.model_dir` directory"""
        loss_path = os.path.join(self.model_dir, "loss.json")
        with open(loss_path, "w") as fp:
            json.dump(self.loss, fp)
