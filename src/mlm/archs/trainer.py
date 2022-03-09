import torch
from tqdm import tqdm

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



    def train(self) -> None:

        print('.... MLM CamemBERT training:')
        print(self.metric_frequency, self.train_metric_steps, self.val_metric_steps)
        super().train()
        print('training done;')
        print()



    def _train_step(self) -> None:

        self.model.train()

        loop = tqdm(self.train_data_loader)
        running_loss = 0
        running_acc = 0
        n_batches = 0

        for i, batch in enumerate(loop):

            n_batches += 1
            self.optimizer.zero_grad()

            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            loss, out = self.model(
                input_ids, attention_mask=attention_mask, labels=labels).to_tuple()

            loss.sum().backward()
            self.optimizer.step()

            
            predictions = torch.argmax(out, dim=-1)
            self.metric.add_batch(
                predictions=predictions.view(-1), references=batch["labels"].view(-1))
            self.tmp_metric.add_batch(
                predictions=predictions.view(-1), references=batch["labels"].view(-1))

            epoch_loss = loss.sum().item()
            epoch_acc = self.tmp_metric.compute()['accuracy']
            running_loss += epoch_loss
            running_acc += epoch_acc

            loop.set_postfix(loss=epoch_loss,
                             acc=epoch_acc)

            if i % self.train_metric_steps == 0:

                self.epoch_loss['train'].append(running_loss / n_batches)
                self.epoch_acc['train'].append(running_acc / n_batches)

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

                loss, out = self.model(
                    input_ids, attention_mask=attention_mask, labels=labels).to_tuple()

            predictions = torch.argmax(out, dim=-1)
            self.metric.add_batch(
                predictions=predictions.view(-1), references=batch["labels"].view(-1))
            self.tmp_metric.add_batch(
                predictions=predictions.view(-1), references=batch["labels"].view(-1))

            epoch_acc = self.tmp_metric.compute()['accuracy']
            epoch_loss = loss.sum().item()

            running_loss += epoch_loss

            loop.set_postfix(loss=epoch_loss, acc=epoch_acc)

            if i % self.val_metric_steps == 0:

                self.epoch_loss['val'].append(running_loss / n_batches)
                self.epoch_acc['val'].append(epoch_acc)

            if i == self.val_steps:

                break
        
        self.loss['val'].append(running_loss / n_batches)
        acc = self.metric.compute()['accuracy']
        self.acc['val'].append(acc)

