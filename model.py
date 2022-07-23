from cmath import exp
import torch
import torch.nn as nn
import torch.utils.data as data_utils
import pdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import joblib

from util_funcs import mse_loss_func, rmse_loss_func, mape_loss_func, r2_loss_func
from util_funcs import validate
from network import ANN_model
from get_train_test import train_test_loader

torch.use_deterministic_algorithms(True)
seed_val = 0
torch.manual_seed(seed_val)
np.random.seed(seed_val)
random.seed(seed_val)

learning_rate = 1e-3
num_epochs = 100

exp_name = "Ash"
train_loader, test_loader, [sc, sct] = train_test_loader(exp_name)
model = ANN_model()

l = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr =learning_rate )

mse_l , rmse_l, mape_l, r2_l = [], [], [], []

for epoch in range(num_epochs):
    for i, (x, y) in enumerate(train_loader):
        # conver x to float tensor
        x = x.float()
        y_pred = model(x.float())
        loss = l(y_pred, y.to(torch.float32))
        optimizer.zero_grad()
        # pdb.set_trace()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f'Epoch: {epoch+1}, train loss: {loss.item()}')
    all_loss, gt, pred = validate(model, test_loader, sct)
    # print(f"test loss: {all_loss}")
    print(f"mse_l: {all_loss[0].item()}, rmse_l: {all_loss[1].item()}, mape_l: {all_loss[2].item()}, r2_l: {all_loss[3].item()}")
    mse_l.append(all_loss[0])
    rmse_l.append(all_loss[1])
    mape_l.append(all_loss[2])
    r2_l.append(all_loss[3])
    
torch.save(model.state_dict(), f"{exp_name}_model.pt")
joblib.dump(sc, "scaler_train.save") 
joblib.dump(sct, "scaler_test.save") 

# plot 4 plots in one graph
fig, ax = plt.subplots(2, 2, figsize=(10,10))
ax[0, 0].plot(mse_l)
ax[0, 0].set_title('MSE')
ax[0, 1].plot(rmse_l)
ax[0, 1].set_title('RMSE')
ax[1, 0].plot(mape_l)
ax[1, 0].set_title('MAPE')
ax[1, 1].plot(r2_l)
ax[1, 1].set_title('R2')
plt.savefig('loss_func.png')
# plt.show()

