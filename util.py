# coding: utf-8

import torch
import numpy as np

def load_data():
    inputs = torch.Tensor([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    targets = torch.Tensor([
        [0],
        [1],
        [1],
        [0]
    ])

    

    return inputs, targets

if __name__ == "__main__":
    load_data()