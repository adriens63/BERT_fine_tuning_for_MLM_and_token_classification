import jsonlines
import re





# ************** tools *********************

def remove_first_and_last_spaces(s: str) -> str:

    expression_spaces = re.compile('^\s*|\s*$')
    return re.sub(expression_spaces, '', s)





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
        #TODO : mettre un codage <$$€> par catégorie de soft skills quand elles seront définies

        masked_sequences = []

        for line in self.lines:
            labels = line['label']
            sequence = line['data']

            for label in labels:
                start, end, label_kind = label
                marked_words = sequence[start:end]
                print(f'marked word : {marked_words}')

                # remove first and last spaces of the beginning
                begining_sequence = remove_first_and_last_spaces(sequence[:start])

                # remove first and last spaces of the end of the sentences
                end_sequence = remove_first_and_last_spaces(sequence[end:])

                n_words = len(marked_words.split())
                
                new_sequence = begining_sequence + ' <£$€>'*n_words + ' ' + end_sequence
                print(new_sequence)
                print()

                masked_sequences.append(new_sequence)
        
        return masked_sequences