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
    
    # YOUR CODE GOES HERE
    pass
