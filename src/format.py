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
    formatter.format_to_TextLine()
    print('done;')
