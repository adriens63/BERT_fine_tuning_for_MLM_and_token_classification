import csv
import tqdm
import torch
from transformers import BertTokenizer
import numpy as np
import numpy.typing as npt
from typing import List






# ***************** loading *******************

class FileLoader:
    
    def __init__(self, path: str) -> None:
        
        self.path = path
        
    
    def load(self) -> None:
        
        with open(self.path) as f:
            csv_reader = csv.reader(f)
            
            categories = next(csv_reader)
            
            self.ds_dict = {}
            for cat in categories:
                self.ds_dict[cat] = []
            
            for s in csv_reader:
                for i, cat in enumerate(categories):
                    self.ds_dict[cat].append(s[i])



class Word2Int:

    def __init__(self) -> None:

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


    def vectorize(self, sequences, max_seq_length, tensor_kind = 'pt'):

        inp = self.tokenizer(sequences, return_tensors = tensor_kind, max_length = max_seq_length, truncation = True, padding = 'max_length')

        return inp




class JobDescriptionDataset(torch.utils.data.Dataset):

    def __init__(self, encodings):

        self.encodings = encodings


    def __get_item__(self, idx):

        return {key : torch.tensor(val[idx]) for key, val in self.encodings.items()}


    def __len__(self):

        return self.encodings['input_ids'].shape[0]



class MaskBlock:

    def __init__(self, frac_msk) -> None:
        
        self.frac_msk = frac_msk

    def get_msk(self, tok) -> npt.NDArray:

        rand = torch.rand(tok.shape)
        
        msk = (rand < self.frac_msk) * (tok != 101) * \
           (tok != 102) * (tok != 0)
    
        return msk

    def get_masked_idx(self, msk) -> List:

        self.selection = []

        for row in range(msk.shape[0]):
            
            self.selection.append(torch.flatten(msk[row].nonzero()).tolist())

        return self.selection


class GetDataset:

    def __init__(self, path: str, max_seq_length: int, frac_msk: float, batch_size: int, shuffle: bool) -> None:

        self.fl = FileLoader(path)
        self.w2i = Word2Int()
        self.m = MaskBlock(frac_msk)

        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.shuffle = shuffle

    def get_sequences(self):
        
        if not hasattr(self.fl, 'ds_dict'):

            self.fl.load()

        sequences = self.fl.ds_dict['lbl_competence']

        return sequences



    def get_ds_ready(self):

        seq = self.get_sequences()

        print('.... Start tokenizing sequences')

        inp = self.w2i.vectorize(seq, self.max_seq_length)

        print('done;')

        inp['labels'] = inp['input_ids'].detach().clone()

        msk = self.m.get_msk(inp['input_ids'])

        idx = self.m.get_masked_idx(msk)

        for row in range(inp['input_ids'].shape[0]):

            inp['input_ids'][row, idx[row]] = 103

        dataset = JobDescriptionDataset(encodings = inp)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size = self.batch_size, shuffle = self.shuffle)

        return dataloader