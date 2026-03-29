"""Tests for pyscivex GPU operations — gpu submodule."""

import pytest
import pyscivex as sv


# ===========================================================================
# DEVICE
# ===========================================================================


class TestDevice:
    def test_create(self):
        dev = sv.gpu.Device()
        assert "Device" in repr(dev)

    def test_info(self):
        dev = sv.gpu.Device()
        info = dev.info()
        assert "name" in info
        assert "backend" in info
        assert "device_type" in info
        assert isinstance(info["name"], str)

    def test_detect_backend(self):
        backend = sv.gpu.detect_backend()
        assert isinstance(backend, str)
        assert len(backend) > 0


# ===========================================================================
# GPU TENSOR
# ===========================================================================


class TestGpuTensor:
    def test_create(self):
        dev = sv.gpu.Device()
        t = sv.gpu.GpuTensor(dev, [1.0, 2.0, 3.0, 4.0], [2, 2])
        assert t.shape() == [2, 2]
        assert t.ndim() == 2
        assert t.numel() == 4

    def test_zeros(self):
        dev = sv.gpu.Device()
        t = sv.gpu.GpuTensor.zeros(dev, [3, 3])
        assert t.shape() == [3, 3]
        assert t.numel() == 9
        data = t.to_list()
        assert all(abs(x) < 1e-6 for x in data)

    def test_to_list(self):
        dev = sv.gpu.Device()
        t = sv.gpu.GpuTensor(dev, [1.0, 2.0, 3.0], [3])
        data = t.to_list()
        assert len(data) == 3
        assert abs(data[0] - 1.0) < 1e-6
        assert abs(data[1] - 2.0) < 1e-6
        assert abs(data[2] - 3.0) < 1e-6

    def test_repr(self):
        dev = sv.gpu.Device()
        t = sv.gpu.GpuTensor(dev, [1.0, 2.0], [2])
        assert "GpuTensor" in repr(t)

    def test_1d(self):
        dev = sv.gpu.Device()
        t = sv.gpu.GpuTensor(dev, [5.0], [1])
        assert t.ndim() == 1
        assert t.numel() == 1


# ===========================================================================
# ELEMENT-WISE OPERATIONS
# ===========================================================================


class TestElementwise:
    def test_add(self):
        dev = sv.gpu.Device()
        a = sv.gpu.GpuTensor(dev, [1.0, 2.0, 3.0, 4.0], [4])
        b = sv.gpu.GpuTensor(dev, [5.0, 6.0, 7.0, 8.0], [4])
        c = sv.gpu.add(a, b)
        data = c.to_list()
        assert abs(data[0] - 6.0) < 1e-5
        assert abs(data[3] - 12.0) < 1e-5

    def test_sub(self):
        dev = sv.gpu.Device()
        a = sv.gpu.GpuTensor(dev, [10.0, 20.0], [2])
        b = sv.gpu.GpuTensor(dev, [3.0, 5.0], [2])
        c = sv.gpu.sub(a, b)
        data = c.to_list()
        assert abs(data[0] - 7.0) < 1e-5
        assert abs(data[1] - 15.0) < 1e-5

    def test_mul(self):
        dev = sv.gpu.Device()
        a = sv.gpu.GpuTensor(dev, [2.0, 3.0], [2])
        b = sv.gpu.GpuTensor(dev, [4.0, 5.0], [2])
        c = sv.gpu.mul(a, b)
        data = c.to_list()
        assert abs(data[0] - 8.0) < 1e-5
        assert abs(data[1] - 15.0) < 1e-5

    def test_div(self):
        dev = sv.gpu.Device()
        a = sv.gpu.GpuTensor(dev, [10.0, 20.0], [2])
        b = sv.gpu.GpuTensor(dev, [2.0, 5.0], [2])
        c = sv.gpu.div(a, b)
        data = c.to_list()
        assert abs(data[0] - 5.0) < 1e-5
        assert abs(data[1] - 4.0) < 1e-5

    def test_add_scalar(self):
        dev = sv.gpu.Device()
        a = sv.gpu.GpuTensor(dev, [1.0, 2.0, 3.0], [3])
        c = sv.gpu.add_scalar(a, 10.0)
        data = c.to_list()
        assert abs(data[0] - 11.0) < 1e-5
        assert abs(data[2] - 13.0) < 1e-5

    def test_mul_scalar(self):
        dev = sv.gpu.Device()
        a = sv.gpu.GpuTensor(dev, [1.0, 2.0, 3.0], [3])
        c = sv.gpu.mul_scalar(a, 3.0)
        data = c.to_list()
        assert abs(data[0] - 3.0) < 1e-5
        assert abs(data[2] - 9.0) < 1e-5

    def test_sub_scalar(self):
        dev = sv.gpu.Device()
        a = sv.gpu.GpuTensor(dev, [10.0, 20.0], [2])
        c = sv.gpu.sub_scalar(a, 5.0)
        data = c.to_list()
        assert abs(data[0] - 5.0) < 1e-5
        assert abs(data[1] - 15.0) < 1e-5


# ===========================================================================
# LINEAR ALGEBRA
# ===========================================================================


class TestLinAlg:
    def test_matmul(self):
        dev = sv.gpu.Device()
        # [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
        a = sv.gpu.GpuTensor(dev, [1.0, 2.0, 3.0, 4.0], [2, 2])
        b = sv.gpu.GpuTensor(dev, [5.0, 6.0, 7.0, 8.0], [2, 2])
        c = sv.gpu.matmul(a, b)
        data = c.to_list()
        assert abs(data[0] - 19.0) < 1e-4
        assert abs(data[1] - 22.0) < 1e-4
        assert abs(data[2] - 43.0) < 1e-4
        assert abs(data[3] - 50.0) < 1e-4

    def test_transpose(self):
        dev = sv.gpu.Device()
        # [[1,2,3],[4,5,6]] → [[1,4],[2,5],[3,6]]
        a = sv.gpu.GpuTensor(dev, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
        t = sv.gpu.transpose(a)
        assert t.shape() == [3, 2]
        data = t.to_list()
        assert abs(data[0] - 1.0) < 1e-5
        assert abs(data[1] - 4.0) < 1e-5
        assert abs(data[2] - 2.0) < 1e-5
        assert abs(data[3] - 5.0) < 1e-5


# ===========================================================================
# REDUCTIONS
# ===========================================================================


class TestReductions:
    def test_sum(self):
        dev = sv.gpu.Device()
        a = sv.gpu.GpuTensor(dev, [1.0, 2.0, 3.0, 4.0], [4])
        s = sv.gpu.sum(a)
        assert abs(s - 10.0) < 1e-4

    def test_mean(self):
        dev = sv.gpu.Device()
        a = sv.gpu.GpuTensor(dev, [2.0, 4.0, 6.0, 8.0], [4])
        m = sv.gpu.mean(a)
        assert abs(m - 5.0) < 1e-4


# ===========================================================================
# ACTIVATIONS
# ===========================================================================


class TestActivations:
    def test_relu(self):
        dev = sv.gpu.Device()
        a = sv.gpu.GpuTensor(dev, [-1.0, 0.0, 1.0, 2.0], [4])
        r = sv.gpu.relu(a)
        data = r.to_list()
        assert abs(data[0]) < 1e-5
        assert abs(data[1]) < 1e-5
        assert abs(data[2] - 1.0) < 1e-5
        assert abs(data[3] - 2.0) < 1e-5

    def test_sigmoid(self):
        dev = sv.gpu.Device()
        a = sv.gpu.GpuTensor(dev, [0.0], [1])
        r = sv.gpu.sigmoid(a)
        data = r.to_list()
        assert abs(data[0] - 0.5) < 1e-5

    def test_tanh(self):
        dev = sv.gpu.Device()
        a = sv.gpu.GpuTensor(dev, [0.0], [1])
        r = sv.gpu.tanh(a)
        data = r.to_list()
        assert abs(data[0]) < 1e-5

    def test_exp(self):
        dev = sv.gpu.Device()
        a = sv.gpu.GpuTensor(dev, [0.0, 1.0], [2])
        r = sv.gpu.exp(a)
        data = r.to_list()
        assert abs(data[0] - 1.0) < 1e-5
        assert abs(data[1] - 2.71828) < 1e-3

    def test_log(self):
        dev = sv.gpu.Device()
        a = sv.gpu.GpuTensor(dev, [1.0, 2.71828], [2])
        r = sv.gpu.log(a)
        data = r.to_list()
        assert abs(data[0]) < 1e-5
        assert abs(data[1] - 1.0) < 1e-3

    def test_negate(self):
        dev = sv.gpu.Device()
        a = sv.gpu.GpuTensor(dev, [1.0, -2.0, 3.0], [3])
        r = sv.gpu.negate(a)
        data = r.to_list()
        assert abs(data[0] + 1.0) < 1e-5
        assert abs(data[1] - 2.0) < 1e-5
        assert abs(data[2] + 3.0) < 1e-5


# ===========================================================================
# UTILITY
# ===========================================================================


class TestUtility:
    def test_fill(self):
        dev = sv.gpu.Device()
        t = sv.gpu.fill(dev, [2, 3], 7.0)
        assert t.shape() == [2, 3]
        data = t.to_list()
        assert all(abs(x - 7.0) < 1e-5 for x in data)


# ===========================================================================
# INTEGRATION (all accessible)
# ===========================================================================


class TestIntegrationAccessible:
    def test_all_accessible(self):
        items = [
            # Classes
            sv.gpu.Device,
            sv.gpu.GpuTensor,
            # Element-wise
            sv.gpu.add,
            sv.gpu.sub,
            sv.gpu.mul,
            sv.gpu.div,
            sv.gpu.add_scalar,
            sv.gpu.mul_scalar,
            sv.gpu.sub_scalar,
            # Linear algebra
            sv.gpu.matmul,
            sv.gpu.transpose,
            # Reductions
            sv.gpu.sum,
            sv.gpu.mean,
            # Activations
            sv.gpu.relu,
            sv.gpu.sigmoid,
            sv.gpu.tanh,
            sv.gpu.exp,
            sv.gpu.log,
            sv.gpu.negate,
            # Utility
            sv.gpu.fill,
            sv.gpu.detect_backend,
        ]
        for item in items:
            assert item is not None
