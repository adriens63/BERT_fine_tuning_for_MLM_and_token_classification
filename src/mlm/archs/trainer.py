import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import os.path as osp
from datasets import load_metric
import json






# ********************* trainer *********************

class Trainer:

    def __init__(
            self,
            device,
            model,
            epochs,
            batch_size,
            #loss_fn,
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
            self.w.add_scalar('loss/train', self.loss['train'][-1], e)
            self.w.add_scalar('loss/val', self.loss['val'][-1], e)

            if self.lr_scheduler is not None:

                self.lr_scheduler.step()

            if self.checkpoint_frequency:

                if not osp.exists(self.ckp_dir):
                    os.makedirs(self.ckp_dir)

                self._save_checkpoint(e)


        self.w.flush()
        self.w.close()

        print('done;')
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

            loop.set_postfix(loss = loss.item())
            
            if i == self.train_steps:

                break
        
        self.loss["train"].append(running_loss / n_batches)
        


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

            if i == self.val_steps:

                    break
            
        self.loss["val"].append(running_loss / n_batches)
        acc = self.metric.compute()
        self.acc["val"].append(acc)



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
        loss_path = osp.join(self.mod_dir, "loss.json")
        with open(loss_path, "w") as fp:
            
            json.dump(self.loss, fp)
