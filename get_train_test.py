import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch.utils.data as data_utils
import torch 


def train_test_loader(exp_name, batch_size=32):

    data_file = f'Data4Dipesh/{exp_name}/{exp_name}_Front_Clean.csv'
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

    x_train=sc.fit_transform(x_train)
    y_train =sct.fit_transform(y_train.values.reshape(-1,1))
    x_test = sc.transform(x_test)
    y_test = sct.transform(y_test.values.reshape(-1,1))

    train_set = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    test_set = data_utils.TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    test_loader = data_utils.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, [sc, sct]

if "__main__" == __name__:
    exp_name = "Ash"
    train_loader, test_loader, [sc, sct] = train_test_loader(exp_name)