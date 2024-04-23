import numpy as np
import pandas as pd
import sklearn as sk
import sklearn.model_selection
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
unique_subject_ids = discharge['subject_id'].unique()

subject_ids_train, subject_ids_val = sk.model_selection.train_test_split(unique_subject_ids, train_size=.8)
subject_ids_val, subject_ids_test = sk.model_selection.train_test_split(subject_ids_val, train_size=.5)

pdb.set_trace()
#np.savez('train_and_val_subject_ids.npz', subject_ids_train=subject_ids_train,
#                                          subject_ids_val=subject_ids_val,
#                                          subject_ids_test=subject_ids_test)

discharge_train = discharge.query('subject_id in @subject_ids_train')
discharge_val = discharge.query('subject_id in @subject_ids_val')
discharge_test = discharge.query('subject_id in @subject_ids_test')

hadm_ids_train = discharge_train['hadm_id'].values
hadm_ids_val = discharge_val['hadm_id'].values

 42 np.savez('train_val_test_hadm_ids.npz', hadm_ids_train=hadm_ids_train, hadm_ids_val=hadm_ids_val,
 43                                         hadm_ids_test=hadm_ids_test)
 44 
 45 
