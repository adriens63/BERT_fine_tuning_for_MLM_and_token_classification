# BERT pre-training and fine-tuning

  

## Presentation

The goal of this project is twofold:

  

* Retrieve skills and soft-skills from french textual job descriptions.

  

* Estimate the 'value' of each skill.

  
  

To achieve this, a BERT (CamemBERT) model is firstly pre-trained on 200K job descriptions for masked language modeling, and then fine-tuned on 1,2K hand labeled job descriptions for token classification. Each token receive 0 if it is not related to a skill, and 1 otherwise.  

The main tools used are the torch and transformers libraries, with Trainers implemented from sratch to practice.

  
  

## Requirements

  

Install the following requirements in your docker container or conda env:

```

pip install -r requirements.txt

```

(There might be some to delete or change the version depending on your container)

  
  

## Pre-training

  

The pre-training time is about 7 hours per epochs with two nvidia TESLA (14GB each) gpus.

  

First adapt the config to your needs (path to data, batch size...) in the

```

./src/mlm/config/config.yml

```

Folder.

  

To launch the pre-training, type this command in your terminal:

```

python -m src.mlm.train --config ./src/mlm/config/config.yml

```

or

```

python -W ignore -m src.mlm.train --config ./src/mlm/config/config.yml

```

  

To see the results and metrics with tensorboard, type:

```

tensorboard --logdir=./src/mlm/weights/pre-trained_bert_1/log_dir/ --port=8012

```

  
  

# Fine-tuning

  

First adapt the config to your needs (path to data, batch size...) in the

```

./src/mlm/config/config.yml

```

Folder.

  

To launch the fine-tuning, type this command in your terminal:

```

python -m src.token_classification.train --config ./src/token_classification/config/config.yml

```

or

```

python -W ignore -m src.token_classification.train --config ./src/token_classification/config/config.yml

```

  

To see the results and metrics with tensorboard:

```

tensorboard --logdir=./src/token_classification/weights/fine_tuned_bert_1/log_dir/ --port=8012

```