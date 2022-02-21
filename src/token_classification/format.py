import os.path as osp
import argparse
import yaml

from src.token_classification.archs.data_formatter import *





# ********************* launch formating ***********************
# cmd to launch : python -m src.token_classification.format

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description = 'formatting for labeling')
    parser.add_argument('--config', type=str, required=True, help='path to yaml config')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    asigning_variables(config)

    print('.... Start formatting')
    path = osp.join(config['path'], config['offres'])
    yaml_path = osp.join(config['path'], config['yaml'])
    formatter = Formatter(path, yaml_path)
    formatter.generate_name()
    formatter.load()
    formatter.sort_desc()
    formatter.format_to_jsonl_in_proportions(n_desc = config['n_sequences'])
    print('done;')
    print()
    print('/!\ Be careful to change the owner of the file before pasting it in doccano with the following command : sudo chown <user> <file>')
