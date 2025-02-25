"""Submission for exercise sheet 1

SUBMIT this file as submission_<STUDENTID>.py where
you replace <STUDENTID> with your student ID, e.g.,
submission_1234567.py
"""
import torch
from einops import rearrange

# Exercise 1.1
def assignment_ex1() -> torch.Tensor:
    tensor = torch.arange(16 * 3 * 32 * 32)
    tensor_float = tensor.to(torch.float32)
    tensor_view = tensor_float.view(16,3,32,32)
    return tensor_view

# Exercise 1.2
def assignment_ex2() -> torch.Tensor:
    T0=torch.arange(16*3*3, dtype=torch.float32).view(16,3,3)
    T1=T0*3
    mul=torch.matmul(T0, T1)
    # summed = mul.sum(dim=1).sum(dim=1)
    sum_of_matrices = mul.sum(dim=(1, 2))
    return sum_of_matrices
    
# Exercice 1.3
def assignment_ex3() -> torch.Tensor:
    T0=torch.arange(16*3*3, dtype=torch.float32).view(16,3,3)
    reshaped_tensor = rearrange(T0, 'b h w -> b w h')
    return reshaped_tensor

'''
print(assignment_ex1())
print(assignment_ex2())
print(assignment_ex3())
'''

