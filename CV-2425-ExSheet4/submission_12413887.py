"""Submission for exercise sheet 2
student_id=12413887.py
"""

import torch
import torch.nn as nn
from typing import Callable
from torch.nn.functional import relu

# Exercise 4.1
class MLP(nn.Module):
    def __init__(self, hidden_channels,out_channels):
        super(MLP, self).__init__()
        self.h=hidden_channels
        self.o=out_channels
        
        self.f1 = nn.Linear(3, self.h,bias=True)
        self.f2=nn.Linear(self.h,self.h,bias=True)
        self.f3=nn.Linear(self.h,self.o,bias=False)
   
    def forward(self, x):
        a=self.f1(x)
        b = relu(a)
        f2_x= self.f2(b)
        c=relu(f2_x)
        d=self.f3(c)
        return a,b,c,d

# H = torch.randint(5, 10, (1,)).item()
# C = torch.randint(2, 5, (1,)).item()
# net = MLP(10,3)
# # print(net.h)
# x = torch.rand(16,3)
# net.forward(x)
