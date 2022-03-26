import yaml
import argparse

import src.token_classification.archs.train_fn as t_fn






# ********************* launch training ***********************
# cmd to launch : python -m src.token_classification.train --config ./src/token_classification/config/config.yml
# cmd to visualize : tensorboard --logdir=./src/token_classification/weights/fine_tuned_bert_1/log_dir/ --port=8013

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'bert training')
    parser.add_argument('--config', type=str, required=True, help='path to yaml config')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    t_fn.train(config)