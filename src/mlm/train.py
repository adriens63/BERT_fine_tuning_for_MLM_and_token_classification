import yaml
import argparse

import src.mlm.archs.train_fn as t_fn





# ********************* launch training ***********************
# cmd to launch : python -m src.mlm.train --config ./src/mlm/config/config.yml

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'bert training')
    parser.add_argument('--config', type=str, required=True, help='path to yaml config')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    t_fn.train(config)