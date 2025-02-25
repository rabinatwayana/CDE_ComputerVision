from otter.test_files import test_case
import torch

OK_FORMAT = False

name = "Exercise 1.2"
points = 2

@test_case(points=2)    
def test_2(assignment_ex2, env):
    A = torch.load('assets/ex2_T.pt', weights_only=False)
    assert (A-env['assignment_ex2']()).norm() < 1e-6