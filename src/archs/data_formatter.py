import tqdm
import jsonlines
import yaml
from math import ceil

from src.archs.data_loader import FileLoader





# ****************** constants ****************

# TODO mettre ça dans une config
PATH = './../statapp/data/'
OFFRES = 'sample_offres.csv'
CV = 'sample_cv.csv'
YAML = 'n_per_cat.yaml'
LBL_DESC_OFFRES = 'dc_descriptifoffre'
LBL_ROME = 'dc_rome_id'
VERSION = '0.0'
N_SEQUENCES = 1500
N_MEMBRES = 5





# ****************** formatting *************

class Formatter(FileLoader):

    def __init__(self, path: str, yaml_path: str) -> None:

        super().__init__(path)
        self.letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
        self.yaml_path = yaml_path

    def generate_name(self) -> None:

        self.new_path_txt = self.path[:-4] + '_v' + VERSION + '.txt'
        self.new_path_json = self.path[:-4] + '_v' + VERSION

    def format_to_TextLine(self) -> None:
        
        if not hasattr(self, 'ds_dict'):

            self.load()
        
        if not hasattr(self, 'new_path'):

            self.generate_name()

        with open(self.new_path_txt, 'w') as new:
            
            for e in tqdm.tqdm(self.ds_dict[LBL_DESC_OFFRES][:N_SEQUENCES]):
                
                new.write(e + '\n')
    
    def format_to_jsonl(self) -> None:

        if not hasattr(self, 'ds_dict'):

            self.load()
        
        if not hasattr(self, 'new_path'):

            self.generate_name()
        
        items = [{'text' : e, 'label' : []} for e in tqdm.tqdm(self.ds_dict[LBL_DESC_OFFRES][:N_SEQUENCES])]

        with jsonlines.open(self.new_path_json + '.jsonl', 'w') as new:
            new.write_all(items)

    def sort_desc(self) -> None:

        if not hasattr(self, 'ds_dict'):

            self.load()

        self.desc = self.ds_dict[LBL_DESC_OFFRES]
        self.rome = self.ds_dict[LBL_ROME]
        n = len(self.rome)


        self.sorted_desc = {letter : [] for letter in self.letters}
        self.proportions = {letter : 0. for letter in self.letters}

        for idx in tqdm.tqdm(range(len(self.rome))):

            for letter in self.letters:

                if self.rome[idx][0] == letter:
                    self.sorted_desc[letter].append(self.desc[idx])
                    break

        for k in self.sorted_desc.keys():
            self.proportions[k] = len(self.sorted_desc[k]) / n
            print(f'proportion de {k}: {self.proportions[k]}')
        print()

    def format_to_jsonl_in_proportions(self, n_desc) -> None:

        if not hasattr(self, 'proportions'):

            self.sort_desc()
        
        if not hasattr(self, 'new_path'):

            self.generate_name()
        
        n_per_cat = {letter : ceil(self.proportions[letter] * n_desc) for letter in self.letters }
        labl_desc = []

        for letter in self.letters:    

            labl_desc += [{'text' : e, 'label' : []} for e in tqdm.tqdm(self.sorted_desc[letter][:n_per_cat[letter]]) ]


        spliter = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)]

        parts = spliter(labl_desc, n_desc // N_MEMBRES)

        for membre in range(N_MEMBRES):
            
            with jsonlines.open(self.new_path_json + '_' + str(membre) + '.jsonl', 'w') as new:
                new.write_all(parts[membre])

        
        with open(self.yaml_path, 'w') as f:
            yaml.dump(n_per_cat, f)
        
        print(f'File written in: {self.new_path_json}')

        




        
        
        


