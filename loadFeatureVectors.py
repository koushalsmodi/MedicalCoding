import numpy as np
import pickle

# This program shows how to load the feature vectors and hadm_ids
# which are saved in numpy arrays by createFeatureVectors.py.

data = np.load('feature_vectors_and_hadm_ids.npz')
feature_vectors = data['feature_vectors']
hadm_ids = data['hadm_ids']

filename = 'target_vectors.pkl'
with open(filename, 'rb') as f:
    target_vectors = pickle.load(f)

print('done loading target vectors')






