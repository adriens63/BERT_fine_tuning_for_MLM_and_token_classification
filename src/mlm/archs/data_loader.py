import csv
import tqdm
import torch
from transformers import CamembertTokenizer
import numpy.typing as npt
from typing import List, Dict
from torch.utils.data import  DataLoader

from tools.timer import timeit






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
            
            for s in tqdm.tqdm(csv_reader):

                for i, cat in enumerate(categories):

                    self.ds_dict[cat].append(s[i])




class Word2Int:

    def __init__(self) -> None:

        #self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenizer = CamembertTokenizer.from_pretrained('camembert-base')


    @timeit
    def vectorize(self, sequences: List[str], max_seq_length: int, tensor_kind: str = 'pt') -> Dict[str, torch.Tensor]:

        inp = self.tokenizer(sequences, return_tensors = tensor_kind, max_length = max_seq_length, truncation = True, padding = 'max_length')

        return inp




class JobDescriptionDataset(torch.utils.data.Dataset):

    def __init__(self, encodings: Dict[str, torch.Tensor]) -> None:

        self.encodings = encodings



    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:

        return {key : val[idx].clone().detach() for key, val in self.encodings.items()}



    def __len__(self) -> int:

        return self.encodings['input_ids'].shape[0]




class MaskBlock:

    def __init__(self, frac_msk: float) -> None:
        
        self.frac_msk = frac_msk



    def get_msk(self, tok: torch.Tensor) -> npt.NDArray:
        """[summary]
        'True' are masked
        We don't want to mask padding token nor first or last token

        Args:
            tok (Tensor): Tensor of tokenized sentences

        Returns:
            npt.NDArray: The mask
        """

        rand = torch.rand(tok.shape)
        
        msk = (rand < self.frac_msk) * (tok != 5) * \
           (tok != 6) * (tok != 1)
    
        return msk



    def get_masked_idx(self, msk: npt.NDArray) -> List:

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



    def get_sequences(self) -> List[str]:
        
        if not hasattr(self.fl, 'ds_dict'):

            self.fl.load()

        sequences = self.fl.ds_dict['dc_descriptifoffre']

        return sequences



    def get_ds_ready(self) -> DataLoader:

        seq = self.get_sequences()

        print('.... Start tokenizing sequences')
        inp = self.w2i.vectorize(seq, self.max_seq_length)
        print('done;')
        print()


        print('.... Start masking')
        inp['labels'] = inp['input_ids'].detach().clone()

        msk = self.m.get_msk(inp['input_ids'])
        idx = self.m.get_masked_idx(msk)

        for row in range(inp['input_ids'].shape[0]):

            inp['input_ids'][row, idx[row]] = 103
        print('done;')
        print()


        dataset = JobDescriptionDataset(encodings = inp)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = self.batch_size, shuffle = self.shuffle)

        return dataloader