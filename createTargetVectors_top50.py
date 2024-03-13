import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import timeit
import pdb

data_dir = '/home/ksmodi/data/'

discharge = pd.read_csv(data_dir + 'discharge.csv')
diagnoses = pd.read_csv(data_dir + 'diagnoses_icd.csv')
diagnoses10 = diagnoses.query('icd_version == 10')
# diagnoses10['parent_code'] = diagnoses10['icd_code'].apply(lambda s:s[0])

counts = diagnoses10['icd_code'].value_counts()
K = 50
top_K = list(counts[:K].index)

code_to_index = {top_K[k]:k for k in range(K)}

target_vectors = {}

time_start = timeit.default_timer()
for i in range(len(diagnoses10)):
    if i % 5000 == 0: print(i)
    hadm_id = diagnoses10.iloc[i]['hadm_id']
    hadm_id = str(hadm_id)
    code = diagnoses10.iloc[i]['icd_code']
    if hadm_id not in target_vectors:
        target_vectors[hadm_id] = np.zeros(K)
    if code in code_to_index:
        target_vectors[hadm_id][code_to_index[code]] = 1.0 

time_stop = timeit.default_timer()
time_elapsed = time_stop - time_start
print('time_elapsed is: ', time_elapsed)

# Now save target_vectors
import pickle
filename = 'target_vectors_topK.pkl'
with open(filename, 'wb') as f:
    pickle.dump(target_vectors, f)




