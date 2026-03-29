"""Tests for pyscivex.linalg submodule."""

import math
import pyscivex as sv


class TestLinalgFunctions:
    def test_det(self):
        a = sv.Tensor([[1.0, 2.0], [3.0, 4.0]])
        d = sv.linalg.det(a)
        assert abs(d - (-2.0)) < 1e-10

    def test_inv(self):
        a = sv.Tensor([[1.0, 2.0], [3.0, 4.0]])
        inv = sv.linalg.inv(a)
        identity = a @ inv
        assert abs(identity.get([0, 0]) - 1.0) < 1e-10
        assert abs(identity.get([1, 1]) - 1.0) < 1e-10

    def test_solve(self):
        a = sv.Tensor([[2.0, 1.0], [1.0, 3.0]])
        b = sv.Tensor([5.0, 10.0], [2])
        x = sv.linalg.solve(a, b)
        assert x.shape() == [2]
        # Verify Ax = b
        ax = a.matvec(x)
        assert abs(ax.to_list()[0] - 5.0) < 1e-10
        assert abs(ax.to_list()[1] - 10.0) < 1e-10

    def test_norm(self):
        x = sv.Tensor([3.0, 4.0], [2])
        assert abs(sv.linalg.norm(x) - 5.0) < 1e-10


class TestLU:
    def test_decompose(self):
        a = sv.Tensor([[2.0, 1.0], [4.0, 3.0]])
        lu = sv.linalg.LU.decompose(a)
        l = lu.l()
        u = lu.u()
        assert l.shape() == [2, 2]
        assert u.shape() == [2, 2]

    def test_det(self):
        a = sv.Tensor([[2.0, 1.0], [4.0, 3.0]])
        lu = sv.linalg.LU.decompose(a)
        assert abs(lu.det() - 2.0) < 1e-10

    def test_solve(self):
        a = sv.Tensor([[2.0, 1.0], [1.0, 3.0]])
        lu = sv.linalg.LU.decompose(a)
        b = sv.Tensor([5.0, 10.0], [2])
        x = lu.solve(b)
        assert x.shape() == [2]

    def test_inverse(self):
        a = sv.Tensor([[2.0, 1.0], [1.0, 3.0]])
        lu = sv.linalg.LU.decompose(a)
        inv = lu.inverse()
        identity = a @ inv
        assert abs(identity.get([0, 0]) - 1.0) < 1e-10


class TestQR:
    def test_decompose(self):
        a = sv.Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        qr = sv.linalg.QR.decompose(a)
        q = qr.q()
        r = qr.r()
        assert q.shape()[0] == 3  # m x m or m x n
        assert r.shape()[1] == 2  # n cols

    def test_is_full_rank(self):
        a = sv.Tensor([[1.0, 0.0], [0.0, 1.0]])
        qr = sv.linalg.QR.decompose(a)
        assert qr.is_full_rank()


class TestSVD:
    def test_decompose(self):
        a = sv.Tensor([[1.0, 2.0], [3.0, 4.0]])
        svd = sv.linalg.SVD.decompose(a)
        sv_vals = svd.singular_values()
        assert len(sv_vals) == 2
        assert sv_vals[0] >= sv_vals[1]  # sorted descending

    def test_rank(self):
        a = sv.Tensor([[1.0, 2.0], [2.0, 4.0]])  # rank 1
        svd = sv.linalg.SVD.decompose(a)
        assert svd.rank(1e-10) == 1


class TestEig:
    def test_symmetric(self):
        # Symmetric 2x2: [[2, 1], [1, 2]] has eigenvalues 1 and 3
        a = sv.Tensor([[2.0, 1.0], [1.0, 2.0]])
        eig = sv.linalg.Eig.decompose_symmetric(a)
        vals = sorted(eig.eigenvalues())
        assert abs(vals[0] - 1.0) < 1e-10
        assert abs(vals[1] - 3.0) < 1e-10


class TestCholesky:
    def test_decompose(self):
        # Positive definite: [[4, 2], [2, 3]]
        a = sv.Tensor([[4.0, 2.0], [2.0, 3.0]])
        chol = sv.linalg.Cholesky.decompose(a)
        l = chol.l()
        assert l.shape() == [2, 2]
        # L should be lower triangular
        assert abs(l.get([0, 1])) < 1e-10

    def test_solve(self):
        a = sv.Tensor([[4.0, 2.0], [2.0, 3.0]])
        chol = sv.linalg.Cholesky.decompose(a)
        b = sv.Tensor([6.0, 5.0], [2])
        x = chol.solve(b)
        assert x.shape() == [2]

    def test_log_det(self):
        a = sv.Tensor([[4.0, 2.0], [2.0, 3.0]])
        chol = sv.linalg.Cholesky.decompose(a)
        ld = chol.log_det()
        # det = 4*3 - 2*2 = 8, ln(8) ≈ 2.079
        assert abs(ld - math.log(8.0)) < 1e-10
