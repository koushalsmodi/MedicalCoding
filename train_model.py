import numpy as np
import sklearn as sk
import sklearn.model_selection
import torch
import pickle
import pdb

# This program shows how to load the feature vectors and hadm_ids
# which are saved in numpy arrays by createFeatureVectors.py.

# BE SURE TO CHANGE THIS LINE TO READ IN THE CORRECT FEATURE VECTORS.
print('BE SURE TO READ IN THE CORRECT FEATURE VECTORS!')
#data = np.load('feature_vectors_and_hadm_ids.npz')
data = np.load('feature_vectors_and_hadm_ids_flan-t5-large.npz')
feature_vectors = data['feature_vectors']
hadm_ids = np.int64(data['hadm_ids'])
d = feature_vectors.shape[1]

train_val_test_splits = np.load('train_val_test_hadm_ids.npz')
hadm_ids_train = train_val_test_splits['hadm_ids_train']
hadm_ids_val = train_val_test_splits['hadm_ids_val']
hadm_ids_test = train_val_test_splits['hadm_ids_test']


filename = 'target_vectors.pkl'
with open(filename, 'rb') as f:
    target_vectors = pickle.load(f)

# Let's split feature_vecs and hadm_ids into train, validation, and test sets.
whr_train = np.where(np.isin(hadm_ids, hadm_ids_train))[0]
whr_val = np.where(np.isin(hadm_ids, hadm_ids_val))[0]
whr_test = np.where(np.isin(hadm_ids, hadm_ids_test))[0]

hadm_ids_train = hadm_ids[whr_train]
hadm_ids_val = hadm_ids[whr_val]
hadm_ids_test = hadm_ids[whr_test]

feature_vectors_train = feature_vectors[whr_train]
feature_vectors_val = feature_vectors[whr_val]
feature_vectors_test = feature_vectors[whr_test]

# The below way of splitting the data was wrong because of data leakage.
# feature_vectors_train, feature_vectors_val, hadm_ids_train, hadm_ids_val = \
#         sk.model_selection.train_test_split(feature_vectors, hadm_ids, train_size=.8, random_state=123)

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
        self.dense1 = torch.nn.Linear(d, 1000) # d is the dimension of our feature vectors.
        self.bn1 = torch.nn.BatchNorm1d(1000)
        self.dense2 = torch.nn.Linear(1000, 500)
        self.bn2 = torch.nn.BatchNorm1d(500)
        self.dense3 = torch.nn.Linear(500, 26)

        self.dropout = torch.nn.Dropout(p=.2)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.dense1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.dense2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.dense3(x)
        return x

class SimpleNN_005(torch.nn.Module):

   def __init__(self):
       super().__init__()
       self.dense1 = torch.nn.Linear(d, 400)
       self.dense2 = torch.nn.Linear(400, 200)
       self.dense3 = torch.nn.Linear(200, 100)
       self.dense4 = torch.nn.Linear(100, 50)
       self.dense5 = torch.nn.Linear(50, 26)
       self.relu = torch.nn.ReLU()
       
   def forward(self, x):
       u1 = self.dense1(x)
       v1 = self.relu(u1)
       u2 = self.dense2(v1)
       v2 = self.relu(u2)
       u3 = self.dense3(v2)
       v3 = self.relu(u3)
       u4 = self.dense4(v3)
       v4 = self.relu(u4)
       u5 = self.dense5(v4)

       return u5
   
model = SimpleNN()
# model = SimpleNN_005()
device = torch.device('cuda')
model.to(device)
loss_fun = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=.001)

L_vals_train = []
L_vals_val = []
acc_vals_train = []
acc_vals_val = []
precision_vals_train = []
precision_vals_val = []
recall_vals_train = []
recall_vals_val = []
F1_micro_vals_train = []
F1_micro_vals_val = []
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

    with torch.inference_mode():

        L_train = 0
        num_correct_train = 0
        TP_train = 0
        FP_train = 0
        FN_train = 0
        for x_batch, y_batch in dataloader_train:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(x_batch)
            loss = loss_fun(outputs, y_batch)
            L_train += loss * len(x_batch)

            probabilities = torch.sigmoid(outputs)
            y_pred = torch.round(probabilities)
            num_correct_train += (y_pred == y_batch).sum()
            
            TP_train += torch.sum((y_pred == 1) & (y_batch == 1))
            FP_train += torch.sum((y_pred == 1) & (y_batch == 0))
            FN_train += torch.sum((y_pred == 0) & (y_batch == 1))


        L_train = L_train / len(dataset_train)
        L_train = L_train.item()
        L_vals_train.append(L_train)

        acc_train = num_correct_train / (len(dataset_train) * 26)
        acc_train = acc_train.item()
        acc_vals_train.append(acc_train)

        precision_train = TP_train / (TP_train + FP_train)
        recall_train = TP_train / (TP_train + FN_train)
        F1_micro_train = 2 * precision_train * recall_train / (precision_train + recall_train)
        precision_vals_train.append(precision_train.item())
        recall_vals_train.append(recall_train.item())
        F1_micro_vals_train.append(F1_micro_train.item())
        print(f'L_train is: {L_train}')
        print(f'acc_train is: {acc_train}')
        print(f'precision_train is: {precision_train}')
        print(f'recall_train is: {recall_train}')
        print(f'F1_micro_train is: {F1_micro_train}')

        L_val = 0
        TP_val = 0
        FP_val = 0
        FN_val = 0
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
           
            TP_val += torch.sum((y_pred == 1) & (y_batch == 1))
            FP_val += torch.sum((y_pred == 1) & (y_batch == 0))
            FN_val += torch.sum((y_pred == 0) & (y_batch == 1))


        precision_val = TP_val / (TP_val + FP_val)
        recall_val = TP_val / (TP_val + FN_val)
        F1_micro_val = 2 * precision_val * recall_val/ (precision_val + recall_val)
        precision_vals_val.append(precision_val.item())
        recall_vals_val.append(recall_val.item())
        F1_micro_vals_val.append(F1_micro_val.item())
        

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
    print(f'precision_val is: {precision_val}')
    print(f'recall_val is: {recall_val}')
    print(f'F1_micro_val is: {F1_micro_val}')



L_vals_train = np.array(L_vals_train)
L_vals_val = np.array(L_vals_val)
acc_vals_train = np.array(acc_vals_train)
acc_vals_val = np.array(acc_vals_val)
precision_vals_train = np.array(precision_vals_train)
precision_vals_val = np.array(precision_vals_val)
recall_vals_train = np.array(recall_vals_train)
recall_vals_val = np.array(recall_vals_val)
F1_micro_vals_train = np.array(F1_micro_vals_train)
F1_micro_vals_val = np.array(F1_micro_vals_val)


from datetime import datetime
current_datetime = datetime.now()
datetime_string = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

# The following line of code hasn't been tested yet!
np.savez('scores_' + datetime_string + '.npz', L_vals_train=L_vals_train, L_vals_val = L_vals_val, 
                       acc_vals_train=acc_vals_train, acc_vals_val=acc_vals_val, 
                       precision_vals_train=precision_vals_train, precision_vals_val=precision_vals_val,
                       recall_vals_train=recall_vals_train, recall_vals_val = recall_vals_val,
                       F1_micro_vals_train=F1_micro_vals_train, F1_micro_vals_val=F1_micro_vals_val)















