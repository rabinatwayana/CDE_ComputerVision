from otter.test_files import test_case
import torch

OK_FORMAT = False

name = "Exercise 2.2"
points = 2

@test_case(points=2)
def test_2(assignment_ex2, env):
    inp = torch.tensor([
        [0.0,0.0],
        [0.0,1.0],
        [1.0,0.0],
        [1.0,1.0]], dtype=torch.float32)
    out = env['assignment_ex2'](inp)
    T = torch.load('assets/ex2_T.pt', weights_only=False)
    assert (T == out).all(), "Output values for OR gate differ from expected values!"
    
    
    