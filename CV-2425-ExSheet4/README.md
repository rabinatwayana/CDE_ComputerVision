# Exercise set 4

*All assignments need to be implemented within the function skeletons found in `submission.py`
and you need to hand in this file in the form `submission_<STUDENTID>.py` at the link provided
for this exercise sheet via e-mail.*

### Exercise 4.1

Create a **Multilayer Perceptron (MLP)** that implements the following function (assuming $\phi$ is a ReLU) $f: \mathbb{R}^3 \to \mathbb{R}^o$:

$$ x \mapsto f(x) = C\phi(B\phi(Ax + a)+b)$$

with 

- $A \in \mathbb{R}^{h \times 3}, a \in \mathbb{R}^h$
- $B \in \mathbb{R}^{h \times h}, b \in \mathbb{R}^h$
- $C \in \mathbb{R}^{h \times o}$

using `torch.nn.Linear` and `torch.nn.functional.relu`. In particular, you should implement this within the class (derived from `nn.Module`)

```python
class MLP(nn.Module):
    def __init__(self,
            # <YOUR CODE GOES HERE>
        )
        super(MLP, self).__init__()
        # <YOUR CODE GOES HERE>

    def forward(self, x):
        # <YOUR CODE GOES HERE>
        pass # replace with correct return statement
```

such that you can pass **two** parameters when instantiating the class, i.e.,

```python
net = MLP(10,3)
```

which specify $g$ (*first* parameter) and $o$ (the **second** parameter). In the constructor `__init__`, you can instantiate the appropriately configured `nn.Linear` layers, and the `forward` method should then implement the flow of information. In particular, the `forward` method should return three things:

1. the output of $Ax + a$,
2. the output of $\phi(Ax + a)$,
3. the output of $\phi(B\phi(Ax + a)+b)$ and
4. the output of $C\phi(B\phi(Ax + a)+b)$.

This can be achieved, e.g., via `return a,b,c,d` (assuming the variables `a,b,c,d` hold the corresponding results).

Automatic evaluation will instantiate your `MLP` class with random 
values of $h$ and $o$ and use a random input tensor `x = torch.rand(16,3)`.

*As in the previous exercise sheets, you can evaluate your solution via*

```bash
otter check submission_XXX.py -q t1 # for Exercise 4.1
```
