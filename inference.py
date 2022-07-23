from cProfile import label
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


exp_name = "Ash"
train_loader, test_loader, [sc, sct] = train_test_loader(exp_name)

model = ANN_model()
model.load_state_dict(torch.load("ash_model.pt"))
model.eval()

# test_losses, gt, pred = validate(model, test_loader, sct)
train_losses, gt, pred = validate(model, train_loader, sct)

mse_l=train_losses[0].item()
rmse_l=train_losses[1].item()
mape_l=train_losses[2].item()
r2=train_losses[3].item()


slope, intercept, r_value, p_value, std_err = stats.linregress(gt,pred)

# plt plot a line with slpe and intercept   
x = np.linspace(0, 60, 10000)
y = slope*x + intercept

fig = plt.figure(figsize=(6,6))

plt.plot(x, y, 'r', label='Regression Line',color=[172.0/255, 77/255.0, 47/255.0])
plt.plot(x, x, 'black', linestyle='dashed', label='Prediction = GroundTruth')
plt.scatter(gt, pred, label='data', facecolors='none', edgecolors='black')
plt.xlabel('Ground Truth')
plt.ylabel(f'Prediction = {slope:.2f}*GroundTruth + {intercept:.2f}')
plt.title(f'Training data | R2: {r2:.4f}')
plt.legend(edgecolor='black', framealpha=1)
plt.savefig(f'{exp_name}_training_regression.png')   

# pdb.set_trace()