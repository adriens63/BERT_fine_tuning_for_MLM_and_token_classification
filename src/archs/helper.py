from transformers import  BertForMaskedLM, AdamW





# ****************** helper ******************

def get_model():

    return BertForMaskedLM.from_pretrained('bert-base-uncased')




def get_optimizer_class(name: str):

    if name == 'AdamW':

        return AdamW

    else:
        
        raise ValueError('Optimizer should be in : AdamW')