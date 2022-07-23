import torch
import torch.nn as nn
import torch.utils.data as data_utils
import pdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from loss_funcs import mse_loss_func, rmse_loss_func, mape_loss_func, r2_loss_func

data_file = 'Data4Dipesh/Ash/Ash_Front_Clean.csv'
df = pd.read_csv(data_file)
df = df.drop(columns=['Unnamed: 0'])

L_after = df[df.columns[7]]
L_before = df[df.columns[4]]
a_after = df[df.columns[8]]
a_before = df[df.columns[5]]
b_after = df[df.columns[9]]
b_before = df[df.columns[6]]

# pdb.set_trace()
X = df[df.columns[1:7]]
X = X.drop(columns=['Point_No'])

Y = np.sqrt((L_after-L_before)**2 + (a_after-a_before)**2 + (b_after-b_before)**2)

x_train, x_test, y_train, y_test = train_test_split(X, Y,  test_size=0.2)

sc = MinMaxScaler()
sct = MinMaxScaler()

# pdb.set_trace()

x_train=sc.fit_transform(x_train)
y_train =sct.fit_transform(y_train.values.reshape(-1,1))
x_test = sc.transform(x_test)
y_test = sct.transform(y_test.values.reshape(-1,1))

train_set = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
train_loader = data_utils.DataLoader(train_set, batch_size=32, shuffle=True)
# train_set = data_utils.TensorDataset(torch.from_numpy(x_train.values), torch.from_numpy(y_train.values))

test_set = data_utils.TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
test_loader = data_utils.DataLoader(test_set, batch_size=32, shuffle=False)



class ANN_model(torch.nn.Module):
    def __init__(self):
        super(ANN_model, self).__init__()
        self.linear1 = torch.nn.Linear(5, 10)  
        # self.linear2 = torch.nn.Linear(15, 10)  
        self.linear2 = torch.nn.Linear(10, 1)  

    def forward(self, x):
        x = self.linear1(x)
        # x = torch.rel(x)
        x = torch.relu(x)
        x = self.linear2(x)
        # x = torch.relu(x)
        # x = self.linear3(x)
        return x
        

model = ANN_model()

learning_rate = 0.0001
l = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr =learning_rate )

mse_l , rmse_l, mape_l, r2_l = [], [], [], []

num_epochs = 500

def validate():
    mse_l , rmse_l, mape_l, r2_l = 0, 0,0,0
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x = x.float()
            y_pred = model(x.float())
            # loss += l(y_pred, y.to(torch.float32))
            y_pred = sct.inverse_transform(y_pred.detach().numpy())
            y = sct.inverse_transform(y.detach().numpy())
            mse_l += mse_loss_func(torch.tensor(y_pred), torch.tensor(y).to(torch.float32))
            rmse_l += rmse_loss_func(torch.tensor(y_pred), torch.tensor(y).to(torch.float32))
            mape_l += mape_loss_func(torch.tensor(y_pred), torch.tensor(y).to(torch.float32))
            r2_l += r2_loss_func(torch.tensor(y_pred), torch.tensor(y).to(torch.float32))

    return mse_l/len(test_loader), rmse_l/len(test_loader), mape_l/len(test_loader), r2_l/len(test_loader)


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
    all_loss = validate()
    # print(f"test loss: {all_loss}")
    print(f"mse_l: {all_loss[0].item()}, rmse_l: {all_loss[1].item()}, mape_l: {all_loss[2].item()}, r2_l: {all_loss[3].item()}")
    mse_l.append(all_loss[0])
    rmse_l.append(all_loss[1])
    mape_l.append(all_loss[2])
    r2_l.append(all_loss[3])
    

    # print(f'Test loss: {loss.item()}')
    # print(f'MSE: {torch.mean(torch.stack(mse_l)).item()}')
    # print(f'RMSE: {torch.mean(torch.stack(rmse_l)).item()}')
    # print(f'MAPE: {torch.mean(torch.stack(mape_l)).item()}')
    # print(f'R2: {torch.mean(torch.stack(r2_l)).item()}')

# plot all the loss functions in one graph

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


# plt.plot(mse_l, label='MSE')
# plt.plot(rmse_l, label='RMSE')
# plt.plot(mape_l, label='MAPE')
# plt.plot(r2_l, label='R2')
# plt.legend()
