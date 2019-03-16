# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim

from model import Net
from util import load_data

if __name__ == "__main__":
    
    inputs, targets = load_data()

    net = Net()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    for epoch in range(5000):

        outputs = net(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 100 == 0:
            print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, 5000, loss.item()))

    y_pred = net(inputs)

    print(y_pred)
    print(targets)
