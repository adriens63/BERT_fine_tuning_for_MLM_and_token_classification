from transformers import  AdamW
from transformers.modeling_camembert import CamembertForMaskedLM





# ****************** helper ******************

def get_model():

    return CamembertForMaskedLM.from_pretrained('bert-base-uncased')




def get_optimizer_class(name: str):

    if name == 'AdamW':

        return AdamW

    else:
        
        raise ValueError('Optimizer should be in : AdamW')