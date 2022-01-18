from src.archs.data_formatter import *
import os.path as osp


if __name__ == '__main__':
    
    print('beginning formatting...')
    path = osp.join(PATH, OFFRES)
    formatter = Formatter(path)
    formatter.generate_name()
    formatter.load()
    formatter.format_to_TextLine()
    print('formatting done;')
