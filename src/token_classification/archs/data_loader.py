import tqdm
import torch
from transformers import CamembertTokenizer
from typing import List, Dict
from torch.utils.data import  DataLoader

from src.mlm.tools.timer import timeit






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

    def __init__(self, encodings: Dict[str, torch.Tensor], selection: List[List[int]]) -> None:

        self.encodings = encodings
        self.selection = selection



    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:

        out = {key : val[idx].clone().detach() for key, val in self.encodings.items()}
        lbl = torch.zeros(size = out['input_ids'].shape, dtype = torch.long)

        lbl[idx, self.selection[idx]] = 1

        out['labels'] = lbl

        return out



    def __len__(self) -> int:

        return self.encodings['input_ids'].shape[0]




class GetDataset:

    def __init__(self, path: str, max_seq_length: int, batch_size: int, shuffle: bool) -> None:

        self.fl = FileLoader(path)
        self.w2i = Word2Int()

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

        dataset = JobDescriptionDataset(encodings = inp)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = self.batch_size, shuffle = self.shuffle)

        return dataloader