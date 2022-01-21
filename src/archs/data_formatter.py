from src.archs.data_loader import FileLoader
import tqdm
import jsonlines





# ****************** constants ****************

# TODO mettre ça dans une config
PATH = './../statapp/data/'
OFFRES = 'sample_offres.csv'
CV = 'sample_cv.csv'
LBL_DESC_OFFRES = 'dc_descriptifoffre'
VERSION = '0.3'
N_SEQUENCES = 50





# ****************** formatting *************

class Formatter(FileLoader):

    def __init__(self, path: str) -> None:

        super().__init__(path)

    def generate_name(self) -> None:

        self.new_path_txt = self.path[:-4] + '_v' + VERSION + '.txt'
        self.new_path_json = self.path[:-4] + '_v' + VERSION + '.jsonl'

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

        with jsonlines.open(self.new_path_json, 'w') as new:
            new.write_all(items)




        




        
        
        


