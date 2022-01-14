import src.archs.trainer as t
import src.archs.data_loader as dl
import src.archs.helper as h






# **************** training *******************

def train(config):

    bert = h.get_model()

    optimizer_class = h.get_optimizer_class(config['optimizer'])
    optimizer = optimizer_class(bert.parameters(), lr = config['learning_rate'])

    train_data_loader = dl.GetDataset(config['train_path'], 
                            config['max_seq_length'], 
                            config['frac_msk'], 
                            config['batch_size'], 
                            config['suffle'])
    val_data_loader = None

    trainer = t.Trainer(device = config['device'],
                        model = bert,
                        epochs = config['epochs'],
                        batch_size = config['batch_size'],
                        loss_fn = None,
                        optimizer = optimizer,
                        lr_scheduler = None,
                        train_data_loader = train_data_loader,
                        train_steps = config['train_step'],
                        val_data_loader = None,
                        val_steps = config['val_step'],
                        #checkpoint_frequency = config['checkpoint_frequency'],
                        model_name = config['model_name'],
                        weights_path = config['weights_path'],
                        #log_dir = config['log_path']
                        )
    
    trainer.train()
    trainer.save_model()

    print('bert saved to directory: ', config['weights_path'])
