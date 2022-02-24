import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import os
import os.path as osp
from datasets import load_metric
import json


NON_LBL_TOKEN = -100 #TODO: put these tokens in file
MAX_GRAD_NORM = 10


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
        #self.loss_fn = loss_fn
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
        self.log_dir = weights_path + model_name + '/log_dir/'
        self.ckp_dir = weights_path + model_name + '/ckp_dir/'

        self.metric = load_metric('accuracy')
        self.loss = {"train": [], "val": []}
        self.acc = {"train": [], "val": []}
        self.w = SummaryWriter(log_dir = self.log_dir )



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
            self.w.add_scalar('loss/train', self.loss['train'][-1])
            self.w.add_scalar('loss/val', self.loss['val'][-1])

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if self.checkpoint_frequency:
                    self._save_checkpoint(e)


        self.w.flush()
        self.w.close()

        print('done;')
        print()



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

            loss, y_pred_logits = self.model(input_ids = ids, attention_mask = msk, labels = lbl)
            
            # loss
            running_loss += loss.item()

            # accuracy
            #TODO : même pas besoin de flatten
            flatten_lbl = lbl.view(-1) # from [b, seq_length] to [b * seq_length], we put all the lbl together to compare all at once
            
            flatten_logits = y_pred_logits.view(-1) # from [b, seq_length, n_lbl] to [b * seq_length, n_lbl], we put all the predicted lbl together
            flatten_pred = torch.argmax(flatten_logits, axis = 1) # compute argmax along last axis to get shape [b, seq_lenght]

            ## keeping only real lbls to perform comparison
            msk_unactive_lbl = flatten_lbl != NON_LBL_TOKEN

            flatten_real_lbl = torch.masked_select(flatten_lbl, mask = msk_unactive_lbl)
            flatten_real_pred = torch.masked_select(flatten_pred, mask = msk_unactive_lbl)

            batch_accuracy = (flatten_real_lbl == flatten_pred).sum() / self.batch_size
            running_accuracy += batch_accuracy

            # grad clipping
            torch.nn.utils.clip_grad_norm(parameters = self.model.parameters(), max_norm = MAX_GRAD_NORM)

            loss.backwards()
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
        running_loss = 0

        for i, batch in enumerate(loop):

            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            with torch.no_grad():

                out = self.model(input_ids, attention_mask = attention_mask, labels = labels)

            loss = out.loss

            running_loss += loss.item()

            logits = out.logits
            predictions = torch.argmax(logits, dim= -1)
            self.metric.add_batch(predictions=predictions, references=batch["labels"])

            if i == self.val_steps:

                    break
            
        self.loss["val"].append(running_loss / self.batch_size)
        self.metric.compute() #TODO ca va ou



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

        print('.... Saving model')
        model_path = osp.join(self.weights_path, self.model_name)
        
        if not osp.exists(model_path):
            
            os.makedirs(model_path)

        torch.save(self.model, model_path + '/' + self.model_name + '.pt')

        print('done;')
        print()


    
    def save_loss(self) -> None:

        """Save train/val loss as json file to `self.model_dir` directory"""
        loss_path = osp.join(self.model_dir, "loss.json")
        with open(loss_path, "w") as fp:
            
            json.dump(self.loss, fp)
