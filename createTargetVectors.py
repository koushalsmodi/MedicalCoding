import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import timeit

data_dir = '/home/ksmodi/data/'

discharge = pd.read_csv(data_dir + 'discharge.csv')
diagnoses = pd.read_csv(data_dir + 'diagnoses_icd.csv')
diagnoses10 = diagnoses.query('icd_version == 10')
diagnoses10['parent_code'] = diagnoses10['icd_code'].apply(lambda s:s[0])

letter_to_index = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6,
                   'H':7, 'I':8, 'J':9, 'K':10, 'L':11, 'M':12, 'N':13, 'O':14,
                   'P':15, 'Q':16, 'R':17, 'S':18, 'T':19, 'U':20, 'V':21, 'W':22,
                   'X':23, 'Y':24, 'Z':25}

target_vectors = {}

time_start = timeit.default_timer()
for i in range(len(diagnoses10)):
    hadm_id = diagnoses10.iloc[i]['hadm_id']
    hadm_id = str(hadm_id)
    pc = diagnoses10.iloc[i]['parent_code']
    if hadm_id not in target_vectors:
        target_vectors[hadm_id] = np.zeros(26)
    target_vectors[hadm_id][letter_to_index[pc]] = 1.0 

time_stop = timeit.default_timer()
time_elapsed = time_stop - time_start
print('time_elapsed is: ', time_elapsed)

# Now save target_vectors
import pickle
filename = 'target_vectors.pkl'
with open(filename, 'wb') as f:
    pickle.dump(target_vectors, f)




