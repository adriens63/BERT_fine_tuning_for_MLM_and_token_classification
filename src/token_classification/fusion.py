import jsonlines
import os
import os.path as osp
import argparse

import string
from transformers import CamembertTokenizer
import tqdm
import torch
from transformers import CamembertTokenizer
from typing import List, Dict
from torch.utils.data import DataLoader

from src.tools.timer import timeit






# ************************* fusion jsonlfiles ****************************

def fusion(path):

    print('....Fusionning files')

    path_out = osp.join(path, 'out.jsonl')

    with jsonlines.open(path_out, 'w') as out:

        for file in os.listdir(path):
            if file == 'out.jsonl':
                continue
            print(f'copying file: {file}')
            path_file = osp.join(path, file)

            with jsonlines.open(path_file) as f:
                for line in tqdm.tqdm(f):
                    out.write(line)
    print('done;')
    print()






# ************************* execute ********************

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'fusionning')
    parser.add_argument('--path', type=str, required=True, help='path to file to fusion')
    args = parser.parse_args()

    fusion(args.path)