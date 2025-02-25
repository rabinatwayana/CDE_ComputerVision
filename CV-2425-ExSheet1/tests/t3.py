from otter.test_files import test_case
import torch

OK_FORMAT = False

name = "Exercise 1.3"
points = 2

@test_case(points=2)
def test_3(assignment_ex3, env):
    A = torch.load('assets/ex3_T.pt', weights_only=False)
    assert (A-env['assignment_ex3']()).norm() < 1e-6  
    