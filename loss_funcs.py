import torch
# testing

def mse_loss_func(y_pred, y):
    return torch.mean((y_pred-y)**2)

# root mean squared error
def rmse_loss_func(y_pred, y):
    return torch.sqrt(torch.mean((y_pred-y)**2))

# mean absolute percentage error
def mape_loss_func(y_pred, y):
    return torch.mean(torch.abs(y_pred-y)/y)*100

# r^2 score
def r2_loss_func(y_pred, y):
    return 1 - torch.mean((y_pred-y)**2)/torch.var(y)


    