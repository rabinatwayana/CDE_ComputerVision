# Exercise set 1

*All assignments need to be implemented within the function skeletons found in `submission.py`
and you need to hand in this file in the form `submission_<STUDENTID>.py` at the link provided
for this exercise sheet via e-mail.*

### Exercise 1.1

Create a `torch` 32-bit floating point tensor that holds a sequence of integers from `0` to `16*3*32*32`, convert the tensor to 32-bit floating point (`torch.float32`) and reshape the tensor to shape  `(16,3,32,32)`. Implement these steps in the function `assignment_ex1` and return the tensor.

### Exercise 1.2

Create a `torch` 32-bit floating point tensor `T0`   that holds a sequence of integers from `0` to `16*3*3` and reshape that tensor to shape `(16,3,3)`. Then, create a second tensor `T1` which is `T0` multiplied by 3. Finally, use `torch.matmul` to multiply all 16 3x3 matrices from `T0` with the 16 3x3 matrices in `T1` and return the result where all `3*3` matrices are summed up. Implement the functionality within `assignment_ex2`.

### Exericse 1.3

Create a `torch` 32-bit floating point tensor `T0`   that holds a sequence of integers from `0` to `16*3*3` and reshape that tensor to shape `(16,3,3)`. Return the tensor where we switch all rows and columns of all 3x3 matrices. Implement this functionality in `assignment_ex3` and use `einops.rearrange`.
