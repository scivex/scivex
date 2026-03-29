"""Tests for pyscivex symbolic math — sym submodule."""

import math
import pyscivex as sv


# ===========================================================================
# SYMBOLS & CONSTANTS
# ===========================================================================


class TestSymbols:
    def test_var(self):
        x = sv.sym.var("x")
        assert "x" in str(x)

    def test_constant(self):
        c = sv.sym.constant(3.14)
        assert c.is_const()
        assert abs(c.as_const() - 3.14) < 1e-10

    def test_pi(self):
        p = sv.sym.sym_pi()
        assert p.is_const()
        assert abs(p.as_const() - math.pi) < 1e-10

    def test_e(self):
        e = sv.sym.sym_e()
        assert e.is_const()
        assert abs(e.as_const() - math.e) < 1e-10

    def test_zero_one(self):
        z = sv.sym.sym_zero()
        assert z.is_zero()
        o = sv.sym.sym_one()
        assert o.is_one()

    def test_free_variables(self):
        x = sv.sym.var("x")
        y = sv.sym.var("y")
        expr = x + y
        fv = expr.free_variables()
        assert "x" in fv
        assert "y" in fv


# ===========================================================================
# EXPRESSIONS & OPERATORS
# ===========================================================================


class TestExpressions:
    def test_add(self):
        x = sv.sym.var("x")
        c = sv.sym.constant(2.0)
        expr = x + c
        result = expr.eval({"x": 3.0})
        assert abs(result - 5.0) < 1e-10

    def test_sub(self):
        x = sv.sym.var("x")
        c = sv.sym.constant(1.0)
        expr = x - c
        result = expr.eval({"x": 5.0})
        assert abs(result - 4.0) < 1e-10

    def test_mul(self):
        x = sv.sym.var("x")
        c = sv.sym.constant(3.0)
        expr = x * c
        result = expr.eval({"x": 4.0})
        assert abs(result - 12.0) < 1e-10

    def test_div(self):
        x = sv.sym.var("x")
        c = sv.sym.constant(2.0)
        expr = x / c
        result = expr.eval({"x": 10.0})
        assert abs(result - 5.0) < 1e-10

    def test_neg(self):
        x = sv.sym.var("x")
        expr = -x
        result = expr.eval({"x": 3.0})
        assert abs(result + 3.0) < 1e-10

    def test_pow(self):
        x = sv.sym.var("x")
        c = sv.sym.constant(2.0)
        expr = x ** c
        result = expr.eval({"x": 3.0})
        assert abs(result - 9.0) < 1e-10

    def test_complex_expr(self):
        x = sv.sym.var("x")
        c2 = sv.sym.constant(2.0)
        c1 = sv.sym.constant(1.0)
        # x^2 + 2*x + 1
        expr = x ** c2 + c2 * x + c1
        result = expr.eval({"x": 3.0})
        assert abs(result - 16.0) < 1e-10  # 9 + 6 + 1

    def test_substitute(self):
        x = sv.sym.var("x")
        c = sv.sym.constant(5.0)
        expr = x + sv.sym.constant(1.0)
        substituted = expr.substitute("x", c)
        result = substituted.eval({})
        assert abs(result - 6.0) < 1e-10

    def test_repr(self):
        x = sv.sym.var("x")
        assert len(repr(x)) > 0
        assert len(str(x)) > 0


# ===========================================================================
# MATH FUNCTIONS
# ===========================================================================


class TestMathFunctions:
    def test_sin(self):
        x = sv.sym.var("x")
        expr = sv.sym.sym_sin(x)
        result = expr.eval({"x": 0.0})
        assert abs(result) < 1e-10

    def test_cos(self):
        x = sv.sym.var("x")
        expr = sv.sym.sym_cos(x)
        result = expr.eval({"x": 0.0})
        assert abs(result - 1.0) < 1e-10

    def test_exp(self):
        x = sv.sym.var("x")
        expr = sv.sym.sym_exp(x)
        result = expr.eval({"x": 0.0})
        assert abs(result - 1.0) < 1e-10

    def test_ln(self):
        x = sv.sym.var("x")
        expr = sv.sym.sym_ln(x)
        result = expr.eval({"x": 1.0})
        assert abs(result) < 1e-10

    def test_sqrt(self):
        x = sv.sym.var("x")
        expr = sv.sym.sym_sqrt(x)
        result = expr.eval({"x": 4.0})
        assert abs(result - 2.0) < 1e-10

    def test_tan(self):
        x = sv.sym.var("x")
        expr = sv.sym.sym_tan(x)
        result = expr.eval({"x": 0.0})
        assert abs(result) < 1e-10

    def test_abs(self):
        x = sv.sym.var("x")
        expr = sv.sym.sym_abs(x)
        result = expr.eval({"x": -5.0})
        assert abs(result - 5.0) < 1e-10


# ===========================================================================
# DIFFERENTIATION
# ===========================================================================


class TestDifferentiation:
    def test_diff_constant(self):
        c = sv.sym.constant(5.0)
        d = sv.sym.sym_diff(c, "x")
        assert d.is_zero()

    def test_diff_var(self):
        x = sv.sym.var("x")
        d = sv.sym.sym_diff(x, "x")
        assert d.is_one()

    def test_diff_polynomial(self):
        x = sv.sym.var("x")
        c3 = sv.sym.constant(3.0)
        c2 = sv.sym.constant(2.0)
        # 3*x^2 → 6*x
        expr = c3 * (x ** c2)
        d = sv.sym.sym_diff(expr, "x")
        result = d.eval({"x": 2.0})
        assert abs(result - 12.0) < 1e-10  # 6*2

    def test_diff_sin(self):
        x = sv.sym.var("x")
        expr = sv.sym.sym_sin(x)
        d = sv.sym.sym_diff(expr, "x")
        # d/dx sin(x) = cos(x), at x=0: cos(0) = 1
        result = d.eval({"x": 0.0})
        assert abs(result - 1.0) < 1e-10

    def test_diff_nth(self):
        x = sv.sym.var("x")
        c3 = sv.sym.constant(3.0)
        # x^3 → 3x^2 → 6x → 6
        expr = x ** c3
        d3 = sv.sym.sym_diff(expr, "x", n=3)
        result = d3.eval({"x": 0.0})
        assert abs(result - 6.0) < 1e-10


# ===========================================================================
# INTEGRATION
# ===========================================================================


class TestIntegration:
    def test_integrate_polynomial(self):
        x = sv.sym.var("x")
        c2 = sv.sym.constant(2.0)
        # integrate x^2 dx → x^3/3
        expr = x ** c2
        integral = sv.sym.sym_integrate(expr, "x")
        # Evaluate at x=3: 3^3/3 = 9
        result = integral.eval({"x": 3.0})
        assert abs(result - 9.0) < 1e-10

    def test_definite_integral(self):
        x = sv.sym.var("x")
        c2 = sv.sym.constant(2.0)
        # integral of x^2 from 0 to 3 = 9
        expr = x ** c2
        result = sv.sym.sym_definite_integral(expr, "x", 0.0, 3.0)
        assert abs(result - 9.0) < 1e-10

    def test_integrate_sin(self):
        x = sv.sym.var("x")
        expr = sv.sym.sym_sin(x)
        # integral of sin(x) from 0 to pi = 2
        result = sv.sym.sym_definite_integral(expr, "x", 0.0, math.pi)
        assert abs(result - 2.0) < 1e-5


# ===========================================================================
# SIMPLIFICATION & ALGEBRA
# ===========================================================================


class TestAlgebra:
    def test_simplify(self):
        x = sv.sym.var("x")
        z = sv.sym.constant(0.0)
        expr = x + z  # x + 0 should simplify to x
        simplified = sv.sym.sym_simplify(expr)
        result = simplified.eval({"x": 42.0})
        assert abs(result - 42.0) < 1e-10

    def test_expand(self):
        x = sv.sym.var("x")
        a = sv.sym.constant(1.0)
        b = sv.sym.constant(2.0)
        # x * (1 + 2) → x*1 + x*2
        expr = x * (a + b)
        expanded = sv.sym.sym_expand(expr)
        result = expanded.eval({"x": 5.0})
        assert abs(result - 15.0) < 1e-10

    def test_factor(self):
        x = sv.sym.var("x")
        c2 = sv.sym.constant(2.0)
        c3 = sv.sym.constant(3.0)
        # 2*x + 3*x → factor out x → x*(2+3)
        expr = c2 * x + c3 * x
        factored = sv.sym.sym_factor(expr, x)
        result = factored.eval({"x": 4.0})
        assert abs(result - 20.0) < 1e-10


# ===========================================================================
# SOLVING
# ===========================================================================


class TestSolving:
    def test_solve_linear(self):
        x = sv.sym.var("x")
        c5 = sv.sym.constant(5.0)
        # 2x - 10 = 0 → x = 5
        expr = sv.sym.constant(2.0) * x - sv.sym.constant(10.0)
        solution = sv.sym.sym_solve_linear(expr, "x")
        result = solution.eval({})
        assert abs(result - 5.0) < 1e-10

    def test_solve_quadratic(self):
        x = sv.sym.var("x")
        c2 = sv.sym.constant(2.0)
        # x^2 - 5x + 6 = 0 → x = 2, 3
        expr = x ** c2 - sv.sym.constant(5.0) * x + sv.sym.constant(6.0)
        roots = sv.sym.sym_solve_quadratic(expr, "x")
        assert len(roots) == 2
        values = sorted([r.eval({}) for r in roots])
        assert abs(values[0] - 2.0) < 1e-10
        assert abs(values[1] - 3.0) < 1e-10


# ===========================================================================
# POLYNOMIAL
# ===========================================================================


class TestPolynomial:
    def test_create(self):
        # 1 + 2x + 3x^2
        p = sv.sym.Polynomial([1.0, 2.0, 3.0])
        assert p.degree() == 2
        assert p.coeffs() == [1.0, 2.0, 3.0]

    def test_eval(self):
        p = sv.sym.Polynomial([1.0, 2.0, 3.0])
        # 1 + 2*2 + 3*4 = 1 + 4 + 12 = 17
        assert abs(p.eval(2.0) - 17.0) < 1e-10

    def test_roots(self):
        # x^2 - 5x + 6 = (x-2)(x-3)
        p = sv.sym.Polynomial([6.0, -5.0, 1.0])
        roots = p.roots()
        roots.sort()
        assert abs(roots[0] - 2.0) < 1e-10
        assert abs(roots[1] - 3.0) < 1e-10

    def test_add(self):
        p1 = sv.sym.Polynomial([1.0, 2.0])
        p2 = sv.sym.Polynomial([3.0, 4.0])
        p3 = p1.add(p2)
        assert p3.coeffs() == [4.0, 6.0]

    def test_mul(self):
        # (1+x)(1+x) = 1 + 2x + x^2
        p = sv.sym.Polynomial([1.0, 1.0])
        result = p.mul(p)
        assert result.degree() == 2
        assert abs(result.coeffs()[0] - 1.0) < 1e-10
        assert abs(result.coeffs()[1] - 2.0) < 1e-10
        assert abs(result.coeffs()[2] - 1.0) < 1e-10

    def test_to_expr(self):
        p = sv.sym.Polynomial([1.0, 2.0])
        expr = p.to_expr("x")
        result = expr.eval({"x": 3.0})
        assert abs(result - 7.0) < 1e-10  # 1 + 2*3

    def test_repr(self):
        p = sv.sym.Polynomial([1.0, 2.0, 3.0])
        r = repr(p)
        assert "Polynomial" in r


# ===========================================================================
# TAYLOR SERIES
# ===========================================================================


class TestTaylor:
    def test_taylor_exp(self):
        x = sv.sym.var("x")
        expr = sv.sym.sym_exp(x)
        # Taylor of e^x around 0 to order 4
        t = sv.sym.sym_taylor(expr, "x", 0.0, 4)
        # At x=0.1, e^0.1 ≈ 1.10517...
        result = t.eval({"x": 0.1})
        assert abs(result - math.exp(0.1)) < 0.001

    def test_maclaurin_sin(self):
        x = sv.sym.var("x")
        expr = sv.sym.sym_sin(x)
        t = sv.sym.sym_maclaurin(expr, "x", 5)
        # sin(0.5) ≈ 0.479425...
        result = t.eval({"x": 0.5})
        assert abs(result - math.sin(0.5)) < 0.001


# ===========================================================================
# INTEGRATION (all accessible)
# ===========================================================================


class TestIntegrationAccessible:
    def test_all_accessible(self):
        items = [
            # Classes
            sv.sym.Expr,
            sv.sym.Polynomial,
            # Constructors
            sv.sym.var,
            sv.sym.constant,
            sv.sym.sym_pi,
            sv.sym.sym_e,
            sv.sym.sym_zero,
            sv.sym.sym_one,
            # Math functions
            sv.sym.sym_sin,
            sv.sym.sym_cos,
            sv.sym.sym_tan,
            sv.sym.sym_exp,
            sv.sym.sym_ln,
            sv.sym.sym_sqrt,
            sv.sym.sym_abs,
            # Calculus
            sv.sym.sym_diff,
            sv.sym.sym_integrate,
            sv.sym.sym_definite_integral,
            # Algebra
            sv.sym.sym_simplify,
            sv.sym.sym_expand,
            sv.sym.sym_factor,
            # Solving
            sv.sym.sym_solve_linear,
            sv.sym.sym_solve_quadratic,
            # Taylor
            sv.sym.sym_taylor,
            sv.sym.sym_maclaurin,
        ]
        for item in items:
            assert item is not None
