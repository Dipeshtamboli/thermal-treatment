from cProfile import label
from time import time
import torch.utils.data as data_utils
import random
import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from network import ANN_model
from util_funcs import validate
from get_train_test import train_test_loader
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib



torch.use_deterministic_algorithms(True)
seed_val = 0
torch.manual_seed(seed_val)
np.random.seed(seed_val)
random.seed(seed_val)

exp_name = "oak"

model = ANN_model()
model.load_state_dict(torch.load(f"{exp_name}_model.pt"))
model.eval()

temp, L_before, a_before, b_before, delta_e =  325, 71.1, 6.5, 22.2, 44.15
time_range = np.arange(0, 100, 10) # in seconds
X = np.zeros((len(time_range), 5))

delta_e = np.repeat(np.expand_dims(np.array((delta_e)),0), len(time_range), axis=0)
input = np.repeat(np.expand_dims(np.array((temp, L_before, a_before, b_before)),0), len(time_range), axis=0)
input = np.concatenate((input, np.expand_dims(time_range, 0).T), 1)
X[:, 0] = input[:, 0]
X[:, 1] = input[:, 4]
X[:, 2:] = input[:, 1:4]

scaler_x = joblib.load("scaler_train.save") 
scaler_y = joblib.load("scaler_test.save") 
X = scaler_x.transform(X)
data_set = data_utils.TensorDataset(torch.from_numpy(X), torch.from_numpy(delta_e))
E_orig , predict_E = [], []
for iter, (x, E) in enumerate(data_set):
    y_pred = model(x.float())
    E_orig.append(E.item())
    # pdb.set_trace()
    predict_E.append(scaler_y.inverse_transform(y_pred.detach().numpy().reshape(-1,1)).item())

print(list(zip(time_range, np.array(predict_E) - np.array(E_orig))))
# pdb.set_trace()
# plot E_orig and predict_E
plt.xlabel("Time (s)")
plt.ylabel("colour difference (E)")
plt.xticks(np.arange(len(time_range)), time_range)
plt.plot(E_orig, label="E_orig")
plt.plot(predict_E, label="predict_E")
plt.legend()
# plt.show()
plt.savefig(f"{exp_name}_E_orig_predict_E.png")
