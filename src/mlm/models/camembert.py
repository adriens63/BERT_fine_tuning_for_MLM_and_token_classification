from transformers import  AdamW
from transformers import CamembertForMaskedLM, AdamW






# ****************** helper ******************

def get_model():

    return CamembertForMaskedLM.from_pretrained('camembert-base')




def get_optimizer_class(name: str):

    if name == 'AdamW':

        return AdamW

    else:
        
        raise ValueError('Optimizer should be in : AdamW')