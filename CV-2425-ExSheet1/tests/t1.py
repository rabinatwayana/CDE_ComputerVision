from otter.test_files import test_case
import torch

OK_FORMAT = False

name = "Exercise 1.1"
points = 2

@test_case(points=2)
def test_1(assignment_ex1, env):
    T = torch.load('assets/ex1_T.pt', weights_only=False)
    assert (T-env['assignment_ex1']()).norm() < 1e-6   