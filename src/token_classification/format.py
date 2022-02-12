from src.token_classification.archs.data_formatter import *
import os.path as osp





# ********************* launch formating ***********************
# cmd to launch : python -m src.token_classification.format

if __name__ == '__main__':
    
    print('.... Start formatting')
    path = osp.join(PATH, OFFRES)
    yaml_path = osp.join(PATH, YAML)
    formatter = Formatter(path, yaml_path)
    formatter.generate_name()
    formatter.load()
    formatter.sort_desc()
    formatter.format_to_jsonl_in_proportions(n_desc = N_SEQUENCES)
    print('done;')
    print()
    print('/!\ Be careful to change the owner of the file before pasting it in doccano with the following command : sudo chown <user> <file>')
