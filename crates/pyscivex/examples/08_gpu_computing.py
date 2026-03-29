"""GPU computing — device detection, tensor operations, matmul."""
import pyscivex as sv

# Detect GPU backend
backend = sv.gpu.detect_backend()
print(f"GPU backend: {backend}")

# Create device
dev = sv.gpu.Device()
print(f"Device: {dev.info()}")

# Create GPU tensors
a = sv.gpu.GpuTensor(dev, [1.0, 2.0, 3.0, 4.0], [2, 2])
b = sv.gpu.GpuTensor(dev, [5.0, 6.0, 7.0, 8.0], [2, 2])
print(f"a = {a}")
print(f"b = {b}")

# Element-wise operations
c = sv.gpu.add(a, b)
print(f"a + b = {c.to_list()}")

d = sv.gpu.mul(a, b)
print(f"a * b = {d.to_list()}")

# Matrix multiplication
result = sv.gpu.matmul(a, b)
print(f"a @ b = {result.to_list()}")  # [19, 22, 43, 50]

# Activations
x = sv.gpu.GpuTensor(dev, [-1.0, 0.0, 1.0, 2.0], [4])
print(f"relu({x.to_list()}) = {sv.gpu.relu(x).to_list()}")
print(f"sigmoid = {[round(v, 4) for v in sv.gpu.sigmoid(x).to_list()]}")

# Reductions
data = sv.gpu.GpuTensor(dev, [1.0, 2.0, 3.0, 4.0, 5.0], [5])
print(f"sum = {sv.gpu.sum(data)}")
print(f"mean = {sv.gpu.mean(data)}")

# Scalar operations
scaled = sv.gpu.mul_scalar(a, 2.0)
print(f"a * 2 = {scaled.to_list()}")
