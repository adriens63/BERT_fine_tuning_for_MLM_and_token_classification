import csv





# ***************** loading *******************

class DataLoader:
    
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