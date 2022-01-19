import torch
from tqdm import tqdm
import os
import os.path as osp





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
            #TODO checkpoint_frequency,
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
        #TODO self.checkpoint_frequency = checkpoint_frequency
        self.model_name = model_name
        self.weights_path = weights_path
        #TODO self.log_dir = log_dir


    def train(self) -> None:

        print('.... Start training')

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
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        print('done;')

    def _train_step(self) -> None:
        
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
    
    def save_model(self) -> None:

        print('.... Save model')
        
        model_path = osp.join(self.weights_path, self.model_name)
        
        if not osp.exists(model_path):
            os.makedirs(model_path)

        
        torch.save(self.model, model_path + '/' + self.model_name + '.pt')

        print('done;')