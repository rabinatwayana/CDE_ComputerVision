from otter.test_files import test_case
import torch
from torch.nn.functional import relu

OK_FORMAT = False

name = "Exercise 4.1"
points = 4

@test_case(points=3)
def test_1(MLP, env):
    x = torch.rand(16,3)
    H = torch.randint(5, 10, (1,)).item()
    C = torch.randint(2, 5, (1,)).item()
    
    net = env['MLP'](H,C)
    out = net(x)

    assert out[0].shape == (16,H), 'Wrong output shape for 1st layer!'
    assert torch.norm(out[1]-relu(out[0])) < 1e-6, 'Wrong activation after 1st layer!'
    assert out[2].shape == (16,H), 'Wrong output shape for 2nd layer!'
    assert out[3].shape == (16,C), 'Wrong output shape for 3rd layer!'

@test_case(points=1)
def test_2(MLP, env):
    H = torch.randint(5, 10, (1,)).item()
    C = torch.randint(2, 5, (1,)).item()
    
    net = env['MLP'](H,C)
    s = 0 
    for p in net.parameters():
        s += p.numel()
    
    nparams = H*3 + H + H*H + H + H*C
    assert s == nparams, "Wrong ({} vs {}) number of parameters!".format(s,nparams)
    