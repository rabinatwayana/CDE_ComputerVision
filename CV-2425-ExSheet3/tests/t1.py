from otter.test_files import test_case
import torch

OK_FORMAT = False

name = "Exercise 3.1"
points = 4

@test_case(points=4)
def test_1(assignment_ex1, env):
    inp = torch.load('assets/x.pth', weights_only=False)
    out = env['assignment_ex1'](inp)
    print(out)
    T = torch.load('assets/out.pth', weights_only=False)
    assert (out-T).abs().item() < 1e-6, "Output values differ by more than 1e-6"
    