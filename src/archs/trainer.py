import torch
from tqdm import tqdm
import os




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
            log_dir
            ) -> None:
        
        self.device = device
        self.model = model
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
        self.log_dir = log_dir

    def train(self):

        for e in range(self.epochs):
            self._train_step()
            print(
                "Epoch: {}/{} ".format(
                    e + 1,
                    self.epochs,
                    # self.loss["train"][-1],
                    # self.loss["val"][-1],
                )
            )

            self.lr_scheduler.step()


    def _train_step(self):
        
        loop = tqdm(self.train_data_loader)
        for batch in loop:
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

            loop.set_postfix(loss=loss.item())
            
            # if i == self.train_steps:
            #     break
    
    def save_model(self):

        model_path = os.path.join(self.model_dir, "model.pt")
        torch.save(self.model, model_path)