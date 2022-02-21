import jsonlines
import string
from transformers import CamembertTokenizer
import tqdm
import torch
from transformers import CamembertTokenizer
from typing import List, Dict
from torch.utils.data import DataLoader

from src.mlm.tools.timer import timeit






# **************** ctes ******************
MAX_LENGTH = 512






# ************** tools *********************

def standardize(s: str) -> str:

    d = {}
    for e in string.punctuation:
        d[e] = e + ' '

    translator = str.maketrans(d)

    return s.translate(translator)






# ************** conversion **************

class JsonlLoader:
    
    def __init__(self, path: str) -> None:
        
        self.path = path
        
    

    def load(self) -> None:
        
        self.lines = []

        with jsonlines.open(self.path) as f:
            for line in f:
                self.lines.append(line)
    


    def get_selection(self) -> None:

        masked_sequences = []
        tokenizer = CamembertTokenizer.from_pretrained('camembert-base')

        for line in self.lines:

            labels = line['label']
            sequence = line['data']

            idx = []

            for label in labels:

                start, end, label_kind = label
                marked_words = standardize(sequence[start:end])
                print(f'marked word : {marked_words}')

                # remove first and last spaces of the beginning
                beg_sequence = standardize(sequence[:start])
                
                # tokenizing
                tok_marked_words = tokenizer(marked_words, return_tensors = 'pt', max_length = MAX_LENGTH, truncation = True)['input_ids']
                tok_beg_sequence = tokenizer(beg_sequence, return_tensors = 'pt', max_length = MAX_LENGTH, truncation = True)['input_ids']

                start_marked = tok_beg_sequence.shape[-1] - 2

                for i in range(tok_marked_words.shape[-1] - 2):

                    idx.append(start_marked + i) # les indices du mak sont là dedans


            masked_sequences.append(idx)
        
        return masked_sequences
    


    def get_sequences(self) -> List[str]:

        seq = [line['data'] for line in self.lines]
        
        return seq




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

        self.jl = JsonlLoader(path)
        self.w2i = Word2Int()

        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.shuffle = shuffle



    def get_ds_ready(self) -> DataLoader:

        if not hasattr(self.jl, 'lines'):

            self.jl.load()

        seq = self.jl.get_sequences()
        sel = self.jl.get_selection()

        print('.... Start tokenizing sequences')
        inp = self.w2i.vectorize(seq, self.max_seq_length)
        print('done;')
        print()

        dataset = JobDescriptionDataset(encodings = inp, selection = sel)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = self.batch_size, shuffle = self.shuffle)

        return dataloader