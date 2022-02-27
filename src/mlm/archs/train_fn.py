import torch
import gc
import os.path as osp

import src.mlm.archs.trainer as t
import src.mlm.archs.data_loader as dl
import src.mlm.models.camembert as c
from src.tools.helper import log_config






# *************** emptying cache **********

gc.collect()
torch.cuda.empty_cache()






# **************** training ****************

def train(config) -> None:

    camembert = c.get_model()


    optimizer_class = c.get_optimizer_class(config['optimizer'])
    optimizer = optimizer_class(camembert.parameters(), lr = config['learning_rate'])

    loss_fn_class = c.get_loss_fn_class(config['loss_fn'])
    loss_fn = loss_fn_class()

    get_full_ds = dl.GetDataset(config['train_path'], 
                            config['max_seq_length'], 
                            config['frac_msk'], 
                            config['n_elements'])
    ds = get_full_ds.get_ds_ready()
    
    ds_size = len(ds)
    val_size = int(config['val_frac'] * ds_size)
    train_size = ds_size - val_size

    train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size])

    
    train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size = config['batch_size'], shuffle = config['shuffle'])
    val_dataloader = torch.utils.data.DataLoader(val_ds, batch_size = config['batch_size'], shuffle = config['shuffle'])


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('.... Device is: ', device)
    print('done;')
    print()


    trainer = t.Trainer(device = device,
                        model = camembert,
                        epochs = config['epochs'],
                        batch_size = config['batch_size'],
                        loss_fn = loss_fn,
                        optimizer = optimizer,
                        lr_scheduler = None,
                        patience = config['patience'],
                        train_data_loader = train_dataloader,
                        train_steps = config['train_step'],
                        val_data_loader = val_dataloader,
                        val_steps = config['val_step'],
                        checkpoint_frequency = config['checkpoint_frequency'],
                        model_name = config['model_name'],
                        weights_path = config['weights_path'],
                        )
    
    trainer.train()
    trainer.save_model()
    trainer.classification_report()
    trainer.save_loss()

    log_config(config, osp.join(config['weights_path'], config['model_name']))

    print('CamemBERT saved to directory: ', config['weights_path'])
