"""
Description
This is a Natural Language Processing(NLP) Based App useful for basic NLP concepts such as follows;
+ Tokenization & Lemmatization using Spacy
+ Named Entity Recognition(NER) using SpaCy
This is built with Streamlit Framework, an awesome framework for building ML and NLP tools.
Purpose
To perform basic and useful NLP task with Spacy
"""
# Core Pkgs
import streamlit as st
import os


# NLP Pkgs
import spacy
import torch
from transformers import CamembertTokenizer
import re 


# Import our model
model = torch.load("../token_classification/weights/fine_tuned_bert_2/fine_tuned_bert_2.pt", map_location = torch.device('cuda'))
device = 'cuda'
NON_ID_TOKEN = 1
NUM_LABELS = 2




# For the display on the screen
def translate(phrase, words):
    return re.sub('|'.join(words), lambda x: ''.join(e + '\u0332' for e in x.group()), phrase)




# Function to Analyse Tokens and Lemma
@st.cache
def text_analyzer(my_text):
	nlp = spacy.load('en_core_web_sm')
	docx = nlp(my_text)
	# tokens = [ token.text for token in docx]
	allData = [('"Token":{},\n"Lemma":{}'.format(token.text,token.lemma_))for token in docx ]
	return allData

# Function For Extracting Entities
@st.cache
def entity_analyzer(my_text):
	nlp = spacy.load('en_core_web_sm')
	docx = nlp(my_text)
	tokens = [ token.text for token in docx]
	entities = [(entity.text,entity.label_)for entity in docx.ents]
	allData = ['"Token":{},\n"Entities":{}'.format(tokens,entities)]
	return allData


def main():
	""" NLP Based App with Streamlit """

	# Title
	st.title("Ultimate NLP Application")
	st.subheader("Natural Language Processing for everyone")
	st.markdown("""
    	#### Description
    	+ This is a Natural Language Processing(NLP) Based App useful for basic NLP task
    	Tokenization, Named Entity Recognition (NER). Built with the help of [LekanAkin](https://github.com/lekanakin). Click any of the checkboxes to get started. test
    	""")

	# Entity Extraction
	if st.checkbox("Get the Named Entities of your text"):
		st.subheader("Identify Entities in your text")

		message = st.text_area("Enter Text","Type Here..")
		if st.button("Extract"):
			entity_result = entity_analyzer(message)
			st.json(entity_result)

	# Tokenization
	if st.checkbox("Get the Tokens and Lemma of text"):
		st.subheader("Tokenize Your Text")

		message = st.text_area("Enter Text","Type Here.")
		if st.button("Analyze"):
			nlp_result = text_analyzer(message)
			st.json(nlp_result)

	# Our model
	if st.checkbox("Get the Soft Skills"):
		st.subheader("Identify the Soft Skills in your text")

		message = st.text_area("Enter Text","Type Here.")
		if st.button("Analyze"):
			tokenizer = CamembertTokenizer.from_pretrained('camembert-base')


			inp = tokenizer(message, return_tensors = 'pt', max_length = 256, truncation = True, padding = 'max_length').to(device)
			ids = inp['input_ids'].to(device)
			msk = inp['attention_mask'].to(device)

			with torch.no_grad():
				out = model(ids, attention_mask = msk).to_tuple()
			out = out[0]

			# from [b, seq_length, n_lbl] to [b * seq_length, n_lbl], we put all the predicted lbl together
			flatten_logits = out.view(-1, NUM_LABELS)

			# compute argmax along last axis to get shape [b, seq_lenght]
			flatten_pred = torch.argmax(flatten_logits, axis=1)

			# keeping only real lbls to perform comparison
			msk_unactive_lbl = ids[0] != NON_ID_TOKEN

			flatten_real_pred_old = torch.masked_select(
                					flatten_pred, mask=msk_unactive_lbl)

			flatten_real_pred = torch.zeros(size = flatten_real_pred_old.shape, dtype = torch.int64).to(device)

			# change of index
			for i in range(flatten_real_pred_old.shape[0]):
				if flatten_real_pred_old[i] == 1:
					try:
						flatten_real_pred[i + 1] = 1
					except IndexError:
						print('Problem occured during change of index')

			flatten_real_ids = torch.masked_select(
                ids[0], mask=msk_unactive_lbl)
			
			soft_skills_tokens = torch.masked_select(
                flatten_real_ids, mask=flatten_real_pred != 0)

			soft_skills_string_list = tokenizer.batch_decode(soft_skills_tokens)


			
			st.text(translate(message, soft_skills_string_list))




if __name__ == '__main__':
	main()