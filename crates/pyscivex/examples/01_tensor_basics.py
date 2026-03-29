"""Tensor basics — creation, arithmetic, reshaping, reductions."""
import pyscivex as sv

# Create tensors from nested lists
a = sv.Tensor([[1.0, 2.0], [3.0, 4.0]])
b = sv.Tensor([[5.0, 6.0], [7.0, 8.0]])
print("a =", a)
print("b =", b)

# Arithmetic
print("a + b =", a + b)
print("a * b =", a * b)
print("a @ b =", a @ b)  # matrix multiply

# Creation helpers
zeros = sv.Tensor.zeros([3, 3])
ones = sv.Tensor.ones([2, 4])
eye = sv.Tensor.eye(3)
rng = sv.Tensor.arange(0.0, 10.0, 1.0)
print("eye(3) =", eye)
print("arange =", rng)

# Element-wise math
t = sv.Tensor([1.0, 4.0, 9.0, 16.0])
print("sqrt =", t.sqrt())
print("abs  =", sv.Tensor([-1.0, -2.0, 3.0]).abs())

# Shape manipulation
t = sv.Tensor.arange(0.0, 12.0, 1.0)
r = t.reshape([3, 4])
print("reshaped =", r.shape())
print("transposed =", r.transpose().shape())

# Reductions
data = sv.Tensor([2.0, 4.0, 6.0, 8.0])
print("sum  =", data.sum())
print("mean =", data.mean())
print("min  =", data.min())
print("max  =", data.max())

# Linear algebra
m = sv.Tensor([[2.0, 1.0], [1.0, 3.0]])
print("det =", sv.linalg.det(m))
print("inv =", sv.linalg.inv(m))
