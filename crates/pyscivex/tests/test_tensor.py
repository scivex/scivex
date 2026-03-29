"""Tests for pyscivex.Tensor."""

import math
import pytest
import pyscivex as sv


class TestTensorCreation:
    def test_from_flat_data(self):
        t = sv.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
        assert t.shape() == [2, 2]
        assert t.numel() == 4
        assert t.ndim() == 2

    def test_from_nested_list(self):
        t = sv.Tensor([[1.0, 2.0], [3.0, 4.0]])
        assert t.shape() == [2, 2]
        assert t.to_list() == [1.0, 2.0, 3.0, 4.0]

    def test_from_nested_3d(self):
        t = sv.Tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        assert t.shape() == [2, 2, 2]
        assert t.numel() == 8

    def test_zeros(self):
        t = sv.Tensor.zeros([3, 4])
        assert t.shape() == [3, 4]
        assert t.numel() == 12
        assert t.sum() == 0.0

    def test_ones(self):
        t = sv.Tensor.ones([2, 3])
        assert t.shape() == [2, 3]
        assert t.sum() == 6.0

    def test_full(self):
        t = sv.Tensor.full([2, 3], 7.0)
        assert t.shape() == [2, 3]
        assert t.sum() == 42.0

    def test_eye(self):
        t = sv.Tensor.eye(3)
        assert t.shape() == [3, 3]
        assert t.sum() == 3.0

    def test_arange(self):
        t = sv.Tensor.arange(5)
        assert t.shape() == [5]
        assert t.to_list() == [0.0, 1.0, 2.0, 3.0, 4.0]

    def test_linspace(self):
        t = sv.Tensor.linspace(0.0, 1.0, 5)
        assert t.shape() == [5]
        assert abs(t.to_list()[0] - 0.0) < 1e-10
        assert abs(t.to_list()[-1] - 1.0) < 1e-10

    def test_scalar(self):
        t = sv.Tensor.scalar(42.0)
        assert t.numel() == 1
        assert float(t) == 42.0


class TestTensorOps:
    def test_add_tensor(self):
        a = sv.Tensor([1.0, 2.0, 3.0], [3])
        b = sv.Tensor([4.0, 5.0, 6.0], [3])
        c = a + b
        assert c.to_list() == [5.0, 7.0, 9.0]

    def test_add_scalar(self):
        a = sv.Tensor([1.0, 2.0, 3.0], [3])
        c = a + 10.0
        assert c.to_list() == [11.0, 12.0, 13.0]

    def test_radd_scalar(self):
        a = sv.Tensor([1.0, 2.0, 3.0], [3])
        c = 10.0 + a
        assert c.to_list() == [11.0, 12.0, 13.0]

    def test_sub(self):
        a = sv.Tensor([5.0, 5.0], [2])
        b = sv.Tensor([1.0, 2.0], [2])
        c = a - b
        assert c.to_list() == [4.0, 3.0]

    def test_mul_tensor(self):
        a = sv.Tensor([2.0, 3.0], [2])
        b = sv.Tensor([4.0, 5.0], [2])
        c = a * b
        assert c.to_list() == [8.0, 15.0]

    def test_mul_scalar(self):
        a = sv.Tensor([1.0, 2.0, 3.0], [3])
        c = a * 2.0
        assert c.to_list() == [2.0, 4.0, 6.0]

    def test_truediv(self):
        a = sv.Tensor([10.0, 20.0], [2])
        c = a / 2.0
        assert c.to_list() == [5.0, 10.0]

    def test_neg(self):
        a = sv.Tensor([1.0, -2.0, 3.0], [3])
        c = -a
        assert c.to_list() == [-1.0, 2.0, -3.0]

    def test_matmul(self):
        a = sv.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
        b = sv.Tensor([5.0, 6.0, 7.0, 8.0], [2, 2])
        c = a @ b
        assert c.shape() == [2, 2]
        assert c.to_list() == [19.0, 22.0, 43.0, 50.0]

    def test_pow(self):
        a = sv.Tensor([2.0, 3.0], [2])
        c = a ** 2.0
        assert c.to_list() == [4.0, 9.0]

    def test_dot(self):
        a = sv.Tensor([1.0, 2.0, 3.0], [3])
        b = sv.Tensor([4.0, 5.0, 6.0], [3])
        assert a.dot(b) == 32.0


class TestTensorMath:
    def test_abs(self):
        t = sv.Tensor([-1.0, 2.0, -3.0], [3])
        assert t.abs().to_list() == [1.0, 2.0, 3.0]

    def test_sqrt(self):
        t = sv.Tensor([4.0, 9.0, 16.0], [3])
        assert t.sqrt().to_list() == [2.0, 3.0, 4.0]

    def test_sin_cos(self):
        t = sv.Tensor([0.0], [1])
        assert abs(t.sin().to_list()[0]) < 1e-10
        assert abs(t.cos().to_list()[0] - 1.0) < 1e-10

    def test_exp_ln(self):
        t = sv.Tensor([0.0, 1.0], [2])
        exp_t = t.exp()
        assert abs(exp_t.to_list()[0] - 1.0) < 1e-10
        assert abs(exp_t.to_list()[1] - math.e) < 1e-10
        # ln(e) = 1
        t2 = sv.Tensor([math.e], [1])
        assert abs(t2.ln().to_list()[0] - 1.0) < 1e-10

    def test_floor_ceil_round(self):
        t = sv.Tensor([1.3, 2.7, -0.5], [3])
        assert t.floor().to_list() == [1.0, 2.0, -1.0]
        assert t.ceil().to_list() == [2.0, 3.0, 0.0]

    def test_clamp(self):
        t = sv.Tensor([-5.0, 0.0, 5.0, 10.0], [4])
        c = t.clamp(0.0, 5.0)
        assert c.to_list() == [0.0, 0.0, 5.0, 5.0]

    def test_powf(self):
        t = sv.Tensor([2.0, 3.0], [2])
        assert t.powf(3.0).to_list() == [8.0, 27.0]

    def test_powi(self):
        t = sv.Tensor([2.0, 3.0], [2])
        assert t.powi(2).to_list() == [4.0, 9.0]


class TestTensorReductions:
    def test_sum(self):
        t = sv.Tensor([1.0, 2.0, 3.0, 4.0], [4])
        assert t.sum() == 10.0

    def test_mean(self):
        t = sv.Tensor([2.0, 4.0, 6.0, 8.0], [4])
        assert t.mean() == 5.0

    def test_product(self):
        t = sv.Tensor([2.0, 3.0, 4.0], [3])
        assert t.product() == 24.0

    def test_min_max(self):
        t = sv.Tensor([3.0, 1.0, 4.0, 1.0, 5.0], [5])
        assert t.min() == 1.0
        assert t.max() == 5.0

    def test_sum_axis(self):
        t = sv.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        s = t.sum_axis(0)
        assert s.to_list() == [5.0, 7.0, 9.0]
        s2 = t.sum_axis(1)
        assert s2.to_list() == [6.0, 15.0]


class TestTensorShape:
    def test_reshape(self):
        t = sv.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [6])
        r = t.reshape([2, 3])
        assert r.shape() == [2, 3]
        assert r.to_list() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    def test_flatten(self):
        t = sv.Tensor([[1.0, 2.0], [3.0, 4.0]])
        f = t.flatten()
        assert f.shape() == [4]
        assert f.to_list() == [1.0, 2.0, 3.0, 4.0]

    def test_transpose(self):
        t = sv.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
        tt = t.transpose()
        assert tt.shape() == [2, 2]
        assert tt.to_list() == [1.0, 3.0, 2.0, 4.0]

    def test_permute(self):
        t = sv.Tensor.arange(24).reshape([2, 3, 4])
        p = t.permute([2, 0, 1])
        assert p.shape() == [4, 2, 3]

    def test_unsqueeze(self):
        t = sv.Tensor([1.0, 2.0, 3.0], [3])
        u = t.unsqueeze(0)
        assert u.shape() == [1, 3]

    def test_squeeze(self):
        t = sv.Tensor([1.0, 2.0, 3.0], [1, 3, 1])
        s = t.squeeze()
        assert s.shape() == [3]

    def test_concat(self):
        a = sv.Tensor([1.0, 2.0], [1, 2])
        b = sv.Tensor([3.0, 4.0], [1, 2])
        c = sv.Tensor.concat([a, b], 0)
        assert c.shape() == [2, 2]

    def test_stack(self):
        a = sv.Tensor([1.0, 2.0], [2])
        b = sv.Tensor([3.0, 4.0], [2])
        c = sv.Tensor.stack([a, b], 0)
        assert c.shape() == [2, 2]


class TestTensorIndexing:
    def test_getitem_int(self):
        t = sv.Tensor([[1.0, 2.0], [3.0, 4.0]])
        row = t[0]
        assert row.to_list() == [1.0, 2.0]

    def test_getitem_negative(self):
        t = sv.Tensor([10.0, 20.0, 30.0], [3])
        assert t[-1] == 30.0

    def test_getitem_multi(self):
        t = sv.Tensor([[1.0, 2.0], [3.0, 4.0]])
        assert t[1, 0] == 3.0

    def test_setitem(self):
        t = sv.Tensor([1.0, 2.0, 3.0], [3])
        t[[1]] = 99.0
        assert t.to_list() == [1.0, 99.0, 3.0]

    def test_get_set(self):
        t = sv.Tensor.zeros([2, 3])
        t.set([1, 2], 42.0)
        assert t.get([1, 2]) == 42.0

    def test_slice(self):
        t = sv.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        s = t.slice([[0, 2], [1, 3]])
        assert s.shape() == [2, 2]
        assert s.to_list() == [2.0, 3.0, 5.0, 6.0]

    def test_select(self):
        t = sv.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        row = t.select(0, 1)
        assert row.to_list() == [4.0, 5.0, 6.0]

    def test_index_select(self):
        t = sv.Tensor([10.0, 20.0, 30.0, 40.0, 50.0], [5])
        s = t.index_select(0, [4, 0, 2])
        assert s.to_list() == [50.0, 10.0, 30.0]

    def test_masked_select(self):
        t = sv.Tensor([10.0, 20.0, 30.0, 40.0], [4])
        s = t.masked_select([True, False, True, False])
        assert s.to_list() == [10.0, 30.0]


class TestTensorSort:
    def test_sort(self):
        t = sv.Tensor([3.0, 1.0, 2.0], [3])
        assert t.sort().to_list() == [1.0, 2.0, 3.0]

    def test_argsort(self):
        t = sv.Tensor([3.0, 1.0, 2.0], [3])
        assert t.argsort() == [1, 2, 0]


class TestTensorProtocols:
    def test_len(self):
        t = sv.Tensor([[1.0, 2.0], [3.0, 4.0]])
        assert len(t) == 2

    def test_bool_single(self):
        assert bool(sv.Tensor.scalar(1.0)) is True
        assert bool(sv.Tensor.scalar(0.0)) is False

    def test_bool_multi_raises(self):
        with pytest.raises(ValueError):
            bool(sv.Tensor([1.0, 2.0], [2]))

    def test_float(self):
        assert float(sv.Tensor.scalar(3.14)) == 3.14

    def test_int(self):
        assert int(sv.Tensor.scalar(42.0)) == 42

    def test_iter(self):
        t = sv.Tensor([1.0, 2.0, 3.0], [3])
        vals = list(t)
        assert vals == [1.0, 2.0, 3.0]

    def test_eq(self):
        a = sv.Tensor([1.0, 2.0], [2])
        b = sv.Tensor([1.0, 2.0], [2])
        assert a == b

    def test_tolist_nested(self):
        t = sv.Tensor([[1.0, 2.0], [3.0, 4.0]])
        nested = t.tolist()
        assert nested == [[1.0, 2.0], [3.0, 4.0]]


class TestTensorLinAlg:
    def test_norm(self):
        t = sv.Tensor([3.0, 4.0], [2])
        assert abs(t.norm() - 5.0) < 1e-10

    def test_det(self):
        t = sv.Tensor([[1.0, 2.0], [3.0, 4.0]])
        assert abs(t.det() - (-2.0)) < 1e-10

    def test_inv(self):
        t = sv.Tensor([[1.0, 2.0], [3.0, 4.0]])
        inv = t.inv()
        assert inv.shape() == [2, 2]
        # A @ A^-1 should be identity
        identity = t @ inv
        assert abs(identity.get([0, 0]) - 1.0) < 1e-10
        assert abs(identity.get([0, 1])) < 1e-10

    def test_solve(self):
        # Solve Ax = b where A = [[1, 2], [3, 4]], b = [5, 6]
        a = sv.Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = sv.Tensor([5.0, 6.0], [2])
        x = a.solve(b)
        assert x.shape() == [2]


class TestTensorRepr:
    def test_repr(self):
        t = sv.Tensor([1.0, 2.0], [2])
        r = repr(t)
        assert isinstance(r, str)
        assert len(r) > 0

    def test_strides(self):
        t = sv.Tensor([[1.0, 2.0], [3.0, 4.0]])
        assert t.strides() == [2, 1]

    def test_is_empty(self):
        t = sv.Tensor.zeros([0])
        assert t.is_empty()
        t2 = sv.Tensor([1.0], [1])
        assert not t2.is_empty()
