from src.archs.data_formatter import *
import os.path as osp





# ********************* launch formating ***********************
# cmd to launch : python -m src.format

if __name__ == '__main__':
    
    print('.... Start formatting')
    path = osp.join(PATH, OFFRES)
    formatter = Formatter(path)
    formatter.generate_name()
    formatter.load()
    formatter.format_to_jsonl()
    print('done;')
    print()
    print('/!\ Be careful to change the owner of the file before pasting it in doccano with the following command : sudo chown <user> <file>')
