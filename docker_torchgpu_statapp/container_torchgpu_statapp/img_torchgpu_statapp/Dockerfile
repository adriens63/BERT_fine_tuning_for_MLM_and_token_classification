FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

# apt
RUN apt-get -y update
RUN apt-get install -y git

# packages
COPY requirements.txt requirements.txt
#copy permet de pouvoir executer une cmd avec requirements juste après, car alors le fichier existe
#le dir du container      		
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r requirements.txt
COPY . .


