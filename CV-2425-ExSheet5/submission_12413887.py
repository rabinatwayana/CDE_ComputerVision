"""Submission for exercise sheet 5

SUBMIT this file as submission_<STUDENTID>.py where
you replace <STUDENTID> with your student ID, e.g.,
submission_1234567.py
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader,TensorDataset
"""
DO NOT MODIFY THIS METHOD
"""
def get_data():
    # Load data
    fuel = pd.read_json('assets/fuel.json')
    X = fuel.copy()
    y = X.pop('FE')

    # Preprocess data
    preprocessor = make_column_transformer(
        (StandardScaler(),
        make_column_selector(dtype_include=np.number)),
        (OneHotEncoder(sparse_output=False),
        make_column_selector(dtype_include=object)),
    )
    X = preprocessor.fit_transform(X)
    y = np.log(y)
    
    # Split data into training/testing
    X_trn, X_tst, y_trn, y_tst = train_test_split(
        X, 
        y, 
        test_size=0.33, 
        random_state=42)
    
    # Create PyTorch datasets
    ds_trn = TensorDataset(
        torch.tensor(np.array(X_trn), dtype=torch.float32),
        torch.tensor(np.array(y_trn), dtype=torch.float32))
    ds_tst = TensorDataset(
        torch.tensor(np.array(X_tst), dtype=torch.float32),
        torch.tensor(np.array(y_tst), dtype=torch.float32))
    return ds_trn, ds_tst


# Exercise 5.1
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.f1=nn.Linear(50,18,bias=True)
        self.f2=nn.Linear(18,1,bias=True)
        
    def forward(self, x):
        fn_f1=self.f1(x)
        act = nn.functional.relu(fn_f1)
        fn_f2= self.f2(act)
        return fn_f2
    
def train():
    ds_trn, _ = get_data()
    dl_trn = DataLoader(ds_trn, batch_size=32, shuffle=True)
    model = MLP()

    loss_fn = nn.MSELoss()
    eta=0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=eta)

    for epoch in range(150):
        for x_trn, y_trn in dl_trn:
            optimizer.zero_grad()
            y_hat=model(x_trn)
            loss = loss_fn(y_hat, y_trn.view(-1,1))
            loss.backward()
            optimizer.step()
    return model