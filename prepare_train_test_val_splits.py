import numpy as np
import pandas as pd
import sklearn as sk
import sklearn.model_selection
import pdb

print('hello world !!')
data_dir = '/home/ksmodi/data/'
diagnoses = pd.read_csv(data_dir + 'diagnoses_icd.csv')
diagnoses10 = diagnoses.query('icd_version == 10')

 14 fname = data_dir + 'discharge.csv'
 15 discharge_full = pd.read_csv(data_dir + 'discharge.csv')
 16 
 17 # Here we should throw out the irrelevant doctor's notes.
 18 relevant_hadm_ids = list(diagnoses10['hadm_id'].unique())
 19 discharge = discharge_full[discharge_full['hadm_id'].isin(relevant_hadm_ids)]
 20 discharge['text_length'] = discharge['text'].apply(len)
 21 
 22 discharge_sorted = discharge.sort_values(by='text_length', ascending=False)
 23 
 24 unique_subject_ids = discharge['subject_id'].unique()
 25 
 26 subject_ids_train, subject_ids_val = sk.model_selection.train_test_split(unique_subject_ids, train_size=.8)
 27 subject_ids_val, subject_ids_test = sk.model_selection.train_test_split(subject_ids_val, train_size=.5)
 28 
 29 pdb.set_trace()
 30 #np.savez('train_and_val_subject_ids.npz', subject_ids_train=subject_ids_train,
 31 #                                          subject_ids_val=subject_ids_val,
 32 #                                          subject_ids_test=subject_ids_test)
 33 
 34 discharge_train = discharge.query('subject_id in @subject_ids_train')
 35 discharge_val = discharge.query('subject_id in @subject_ids_val')
 36 discharge_test = discharge.query('subject_id in @subject_ids_test')
 37 
 38 hadm_ids_train = discharge_train['hadm_id'].values
 39 hadm_ids_val = discharge_val['hadm_id'].values
