from transformers import  AdamW
from transformers import CamembertForTokenClassification, AdamW
from torch.nn import CrossEntropyLoss






# ****************** helper ******************

def get_model():

    return CamembertForTokenClassification.from_pretrained('camembert-base')




def get_optimizer_class(name: str):

    if name == 'AdamW':

        return AdamW

    else:
        
        raise ValueError('Optimizer should be in : AdamW')




def get_loss_fn_class(name: str):

    if name == 'CrossEntropyLoss':

        return CrossEntropyLoss

    else:
        
        raise ValueError('Loss_fn should be in : CrossEntropyLoss')