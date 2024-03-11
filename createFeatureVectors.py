import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5EncoderModel
import torch
import timeit
import pdb

print('hello world !!')

data_dir = '/home/ksmodi/data/'
diagnoses = pd.read_csv(data_dir + 'diagnoses_icd.csv')
diagnoses10 = diagnoses.query('icd_version == 10')

fname = data_dir + 'discharge.csv'
discharge_full = pd.read_csv(data_dir + 'discharge.csv')

# Here we should throw out the irrelevant doctor's notes.
relevant_hadm_ids = list(diagnoses10['hadm_id'].unique())
discharge = discharge_full[discharge_full['hadm_id'].isin(relevant_hadm_ids)]
discharge['text_length'] = discharge['text'].apply(len)

discharge_sorted = discharge.sort_values(by='text_length', ascending=False)


df = discharge_sorted[['hadm_id', 'text']]
pdb.set_trace()
tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-small')
model = T5EncoderModel.from_pretrained('google/flan-t5-small')

# tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-base')
# model = T5EncoderModel.from_pretrained('google/flan-t5-base')

N = len(df)
#N = 500
# N = 1000
batch_size = 32
batch_size = 128
num_batches = int(np.ceil(N / batch_size))
chunk_size = 500
embedding_vec_dim = 512

feature_vectors = torch.zeros((N, embedding_vec_dim)) 
hadm_ids = np.zeros(N)
# feature_vectors is a tensor that  will store the feature vectors
# that we compute below (one feature vector for each doctor's note).

device = torch.device('cuda')
model.to(device) # Here we put the model on the GPU

time_start = timeit.default_timer()
# The following for loop creates a feature vector for each doctor's note.
for batch_num in range(num_batches):
    print(f'batch_num is: {batch_num}')
    start_idx = batch_num * batch_size
    end_idx = min((batch_num + 1) * batch_size, N)
    batch_texts = list(df.iloc[start_idx:end_idx]['text'])
    batch_hadms = list(df.iloc[start_idx:end_idx]['hadm_id'])
    # batch_texts = list(df.text[start_idx:end_idx])
    input_ids = tokenizer(batch_texts, return_tensors='pt', padding=True).input_ids
    input_ids = input_ids.to(device)
    # input_ids has shape something like (32, 5250)
    num_cols = input_ids.shape[1] # num_cols is something like 5250
    num_chunks = int(np.ceil(num_cols / chunk_size))

    embedding_vector_means = torch.zeros((len(input_ids), embedding_vec_dim))
    embedding_vector_means = embedding_vector_means.to(device)
    col_start = 0
    for chunk_idx in range(num_chunks):
        col_start = chunk_size * chunk_idx
        col_stop = np.min((col_start + chunk_size, num_cols))

        with torch.no_grad():
            outputs = model(input_ids[:, col_start:col_stop], return_dict=True)
            embedding_vectors = outputs['last_hidden_state'].sum(dim=1)
            embedding_vector_means += embedding_vectors

    embedding_vector_means = (1 / num_cols) * embedding_vector_means

    feature_vectors[start_idx:end_idx, :] = embedding_vector_means
    hadm_ids[start_idx:end_idx] = batch_hadms

time_stop = timeit.default_timer()
time_elapsed = time_stop - time_start

feature_vectors = feature_vectors.numpy()
np.savez('feature_vectors_and_hadm_ids', feature_vectors=feature_vectors, hadm_ids=hadm_ids)
















