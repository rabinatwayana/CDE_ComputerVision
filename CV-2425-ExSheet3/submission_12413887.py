"""Submission for exercise sheet 2
SUBMIT this file as submission_<STUDENTID>.py where
you replace <STUDENTID> with your student ID, e.g.,
submission_1234567.py
"""

import torch
import torch.nn as nn
from typing import Callable

# Exercise 3.1 
def assignment_ex1(x: torch.tensor) ->  torch.tensor:
    A = torch.load('assets/A.pth', weights_only=False)
    w = torch.load('assets/w.pth', weights_only=False)

    lr=0.1

    A.requires_grad = True
    w.requires_grad = True
    # x.requires_grad = True

    elu = nn.ELU()

    for epoch in range(11):
        Ax=torch.matmul(A,x)
        f=torch.dot(w, elu(Ax))
        f.backward()
        with torch.no_grad():
            A.data = (A - lr * A.grad)
            w.data = (w - lr * w.grad)
        A.grad.zero_() #to prevent accumulation of gradient
        w.grad.zero_()
    return f.view(-1)