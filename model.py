# coding: utf-8

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 3) # 3つ目の引数は
        self.fc2 = nn.Linear(3, 1) 

    def forward(self, input):
        output = F.relu(self.fc1(input))
        output = self.fc2(output)
        return output