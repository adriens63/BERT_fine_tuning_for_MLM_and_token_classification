import jsonlines
import re
import string
from transformers import CamembertTokenizer



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
    


    def mask_softskills(self):

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