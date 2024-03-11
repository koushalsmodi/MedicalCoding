import numpy as np
import sklearn as sk
import sklearn.model_selection
import torch
import pickle
import pdb

# This program shows how to load the feature vectors and hadm_ids
# which are saved in numpy arrays by createFeatureVectors.py.

data = np.load('feature_vectors_and_hadm_ids.npz')
feature_vectors = data['feature_vectors']
hadm_ids = np.int64(data['hadm_ids'])


filename = 'target_vectors.pkl'
with open(filename, 'rb') as f:
    target_vectors = pickle.load(f)

feature_vectors_train, feature_vectors_val, hadm_ids_train, hadm_ids_val = \
        sk.model_selection.train_test_split(feature_vectors, hadm_ids, train_size=.8, random_state=123)

class ICD10_Dataset():
    def __init__(self, fv, ids, tv):
        self.feature_vectors = fv
        self.hadm_ids = ids
        self.target_vectors = tv

    def __len__(self):
        return len(self.hadm_ids)

    def __getitem__(self, i):
        x = self.feature_vectors[i]
        hadm_id = str(self.hadm_ids[i])
        y = self.target_vectors[hadm_id]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype = torch.float32)

dataset_train = ICD10_Dataset(feature_vectors_train, hadm_ids_train, target_vectors)
dataset_val = ICD10_Dataset(feature_vectors_val, hadm_ids_val, target_vectors)

dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=64, shuffle=True)
dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=64, shuffle=False)

class SimpleNN(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.dense1 = torch.nn.Linear(512, 26)

    def forward(self, x):
        u = self.dense1(x)
        return u
    
model = SimpleNN()
device = torch.device('cuda')
model.to(device)
loss_fun = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

L_vals_train = []
L_vals_val = []
acc_vals_train = []
acc_vals_val = []
num_epochs = 20
for ep in range(num_epochs):
    print(f'ep is: {ep}')

    for x_batch, y_batch in dataloader_train:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        outputs = model(x_batch)
        loss = loss_fun(outputs, y_batch)
        model.zero_grad()
        loss.backward()
        optimizer.step()

    L_train = 0
    num_correct_train = 0
    for x_batch, y_batch in dataloader_train:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        outputs = model(x_batch)
        loss = loss_fun(outputs, y_batch)
        L_train += loss * len(x_batch)

        probabilities = torch.sigmoid(outputs)
        y_pred = torch.round(probabilities)
        num_correct_train += (y_pred == y_batch).sum()


    L_train = L_train / len(dataset_train)
    L_train = L_train.item()
    L_vals_train.append(L_train)

    acc_train = num_correct_train / (len(dataset_train) * 26)
    acc_train = acc_train.item()
    acc_vals_train.append(acc_train)
    print(f'L_train is: {L_train}')
    print(f'acc_train is: {acc_train}')

    L_val = 0
    num_correct_val = 0
    num_zeros_val = 0
    for x_batch, y_batch in dataloader_val:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        outputs = model(x_batch)
        loss = loss_fun(outputs, y_batch)
        L_val += loss * len(x_batch)

        probabilities = torch.sigmoid(outputs)
        y_pred = torch.round(probabilities)
        num_correct_val += (y_pred == y_batch).sum()

        num_zeros_val += (y_batch == 0).sum()

    L_val = L_val / len(dataset_val)
    L_val = L_val.item()
    L_vals_val.append(L_val)

    acc_val = num_correct_val / (len(dataset_val) * 26)
    acc_val = acc_val.item()
    acc_vals_val.append(acc_val)
    acc_val_simple = num_zeros_val.item() / (len(dataset_val) * 26)
    print(f'L_val is: {L_val}')
    print(f'acc_val is: {acc_val}')
    print(f'accuracy predicting all zeros is: {acc_val_simple}')

















