import torch
import torch.nn as nn
import torch.nn.functional as F

class ANN_model(torch.nn.Module):
    def __init__(self):
        super(ANN_model, self).__init__()
        self.linear1 = torch.nn.Linear(5, 12)  
        # self.linear2 = torch.nn.Linear(15, 10)  
        self.linear2 = torch.nn.Linear(12, 1)  

    def forward(self, x):
        x = self.linear1(x)
        # x = torch.rel(x)
        x = torch.relu(x)
        x = self.linear2(x)
        # x = torch.relu(x)
        # x = self.linear3(x)
        return x
        
if "__main__" == __name__:
    model = ANN_model()
    print(model)
    input = torch.FloatTensor(10, 5)
    # model.load_state_dict(torch.load("ash_model.pt"))
    model.eval()
    print(model(input).shape)    