import torch
import gc

import src.mlm.archs.trainer as t
import src.mlm.archs.data_loader as dl
import src.mlm.models.camembert as c






# *************** emptying cache **********
gc.collect()
torch.cuda.empty_cache()






# **************** training ****************

def train(config) -> None:

    camembert = c.get_model()


    optimizer_class = c.get_optimizer_class(config['optimizer'])
    optimizer = optimizer_class(camembert.parameters(), lr = config['learning_rate'])


    get_ds = dl.GetDataset(config['train_path'], 
                            config['max_seq_length'], 
                            config['frac_msk'], 
                            config['batch_size'], 
                            config['shuffle'])
    train_data_loader = get_ds.get_ds_ready()
    val_data_loader = None


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('.... Device is :', device)
    print('done;')
    print()


    trainer = t.Trainer(device = device,
                        model = camembert,
                        epochs = config['epochs'],
                        batch_size = config['batch_size'],
                        optimizer = optimizer,
                        lr_scheduler = None,
                        train_data_loader = train_data_loader,
                        train_steps = config['train_step'],
                        val_data_loader = val_data_loader,
                        val_steps = config['val_step'],
                        checkpoint_frequency = config['checkpoint_frequency'],
                        model_name = config['model_name'],
                        weights_path = config['weights_path'],
                        )
    
    trainer.train()
    trainer.save_model()

    print('CamemBERT saved to directory: ', config['weights_path'])
