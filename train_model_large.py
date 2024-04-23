  import numpy as np
  import sklearn as sk
  import sklearn.model_selection
  import torch
  import pickle
  import pdb
  7 
  8 # This program shows how to load the feature vectors and hadm_ids
  9 # which are saved in numpy arrays by createFeatureVectors.py.
 10 
 11 # BE SURE TO CHANGE THIS LINE TO READ IN THE CORRECT FEATURE VECTORS.
 12 print('BE SURE TO READ IN THE CORRECT FEATURE VECTORS!')
 13 #data = np.load('feature_vectors_and_hadm_ids.npz')
 14 data = np.load('feature_vectors_and_hadm_ids_flan-t5-large.npz')
 15 feature_vectors = data['feature_vectors']
 16 hadm_ids = np.int64(data['hadm_ids'])
 17 d = feature_vectors.shape[1]
 18 
 19 train_val_test_splits = np.load('train_val_test_hadm_ids.npz')
 20 hadm_ids_train = train_val_test_splits['hadm_ids_train']
 21 hadm_ids_val = train_val_test_splits['hadm_ids_val']
 22 hadm_ids_test = train_val_test_splits['hadm_ids_test']
