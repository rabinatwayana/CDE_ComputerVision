"""Submission for exercise sheet 2

otter check submission_12413887.py -q t1
"""

import torch
import torch.nn as nn
from typing import Callable

# Exercise 2.1 (AND gate)
def assignment_ex1(x: torch.tensor) -> Callable[[torch.tensor], torch.tensor]:
    f = nn.Linear(2, 1)  #defining the linear function that take 2 dim input and return 1 dim output
    # setting weights and bias manually
    with torch.no_grad():
        f.weight[:] = torch.tensor([[1.0, 1.0]])  
        f.bias[:] = torch.tensor([-1.5])  
    output= (f(x) > 0).int()   
    return output
    
# Exercise 2.2 (OR gate)
def assignment_ex2(x: torch.tensor) -> Callable[[torch.tensor], torch.tensor]:
    f = nn.Linear(2, 1,bias=False)
    f.weight.requires_grad=False #disable weight update in training
    # setting weights manually
    f.weight[:] = torch.tensor([[1.0, 1.0]]) 
    # with torch.no_grad():  #alternative of requires_grad
        # f.weight[:] = torch.tensor([[1.0, 1.0]])  
    output= (f(x) > 0).int()   
    return output

'''
To test the function:
x = torch.tensor([1.0, 1.0]) 
print(assignment_ex1(x))
'''