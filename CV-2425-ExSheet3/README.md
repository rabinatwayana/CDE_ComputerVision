# Exercise set 3

*All assignments need to be implemented within the function skeletons found in `submission.py`
and you need to hand in this file in the form `submission_<STUDENTID>.py` at the link provided
for this exercise sheet via e-mail.*

### Exercise 3.1

Implement the following function $f: \mathbb{R}^5 \to \mathbb{R}$

$$ \mathbf{x} \mapsto \langle \mathbf{w}, \phi(\mathbf{A}
\mathbf{x})\rangle, \quad \text{with} \quad \phi(x) = \begin{cases}
x & x > 0 \\
\exp(x)-1 & x \leq 0 
\end{cases}$$

with parameters $\mathbf{A} \in \mathbb{R}^{2 \times 5}$, $\mathbf{w} \in \mathbb{R}^2$. Perform **ten (10)** gradient descent update steps on $\mathbf{A}$ and $\mathbf{w}$ with a step size of 0.1. The $\phi$ function is available as `torch.nn.ELU()` in PyTorch.

In the template code, the matrix $\mathbf{A}$, the vector $\mathbf{w}$ are loaded using `torch.load`. The input $\mathbf{x}$ is passed to the function `assignment_ex1` (where you implement your solution). Return the value of the function $f$ evaluated at the provided $\mathbf{x}$ *after* the ten gradient descent update steps have been performed (so, one scalar value is returned). For automatic evaluation to work properly, and assuming your result is stored in a tensor `out`, please always return `out.view(-1)`.

---

**Hints**: There are essentially two ways you can do this:

*First*, use 
`nn.Linear` to implement $\mathbf{A}\mathbf{x}$ and another `nn.Linear` to implement $\langle \mathbf{w}, \cdot \rangle$. In both cases, the weights of the `nn.Linear` modules need to be set correctly. Also, if you for example 
create an instance of `nn.Linear` as `l = nn.Linear(10,20)`, you can easily reset gradients to zero using `l.zero_grad()`.

*Second*, you can implement the matrix-vector multiplication and the dot product directly, using `torch.matmul` and `torch.dot`, respectively. In case you choose this variant, you can reset gradients via `A.grad.zero_()` for instance.

*As in the first exercise sheet, you can evaluate your solution via*

```bash
otter check submission_XXX.py -q t1 # for Exercise 3.1
```
