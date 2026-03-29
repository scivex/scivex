"""Tests for pyscivex.optim — Optimization & Solvers."""

import math
import pyscivex as sv


# ===========================================================================
# ROOT FINDING
# ===========================================================================


class TestBisection:
    def test_simple_root(self):
        """x^2 - 4 = 0 → root at x=2."""
        r = sv.optim.bisection(lambda x: x**2 - 4, 0.0, 3.0)
        assert r["converged"]
        assert abs(r["root"] - 2.0) < 1e-10

    def test_trig_root(self):
        """sin(x) = 0 near π."""
        r = sv.optim.bisection(lambda x: math.sin(x), 3.0, 3.3)
        assert abs(r["root"] - math.pi) < 1e-10


class TestBrentq:
    def test_cubic_root(self):
        """x^3 - 8 = 0 → root at x=2."""
        r = sv.optim.brentq(lambda x: x**3 - 8, 0.0, 3.0)
        assert r["converged"]
        assert abs(r["root"] - 2.0) < 1e-10


class TestNewton:
    def test_sqrt2(self):
        """x^2 - 2 = 0 → root at √2."""
        r = sv.optim.newton(
            lambda x: x**2 - 2,
            lambda x: 2 * x,
            x0=1.5,
        )
        assert r["converged"]
        assert abs(r["root"] - math.sqrt(2)) < 1e-10


# ===========================================================================
# NUMERICAL INTEGRATION
# ===========================================================================


class TestTrapezoid:
    def test_x_squared(self):
        """∫₀¹ x² dx = 1/3."""
        r = sv.optim.trapezoid(lambda x: x**2, 0.0, 1.0, n=10000)
        assert abs(r["value"] - 1.0 / 3.0) < 1e-6


class TestSimpson:
    def test_x_squared(self):
        """∫₀¹ x² dx = 1/3."""
        r = sv.optim.simpson(lambda x: x**2, 0.0, 1.0, n=100)
        assert abs(r["value"] - 1.0 / 3.0) < 1e-10

    def test_sin(self):
        """∫₀^π sin(x) dx = 2."""
        r = sv.optim.simpson(lambda x: math.sin(x), 0.0, math.pi, n=100)
        assert abs(r["value"] - 2.0) < 1e-6


class TestQuad:
    def test_exp(self):
        """∫₀¹ e^x dx = e - 1."""
        r = sv.optim.quad(lambda x: math.exp(x), 0.0, 1.0)
        assert abs(r["value"] - (math.e - 1)) < 1e-8


# ===========================================================================
# INTERPOLATION
# ===========================================================================


class TestInterp1d:
    def test_linear(self):
        xs = [0.0, 1.0, 2.0, 3.0]
        ys = [0.0, 1.0, 4.0, 9.0]
        result = sv.optim.interp1d(xs, ys, [0.5, 1.5, 2.5])
        assert len(result) == 3
        # Linear interpolation at 0.5 → (0+1)/2 = 0.5
        assert abs(result[0] - 0.5) < 1e-10

    def test_cubic(self):
        xs = [0.0, 1.0, 2.0, 3.0, 4.0]
        ys = [0.0, 1.0, 4.0, 9.0, 16.0]
        result = sv.optim.interp1d(xs, ys, [0.5, 2.5], method="cubic")
        assert len(result) == 2


class TestCubicSpline:
    def test_eval(self):
        xs = [0.0, 1.0, 2.0, 3.0]
        ys = [0.0, 1.0, 4.0, 9.0]
        spline = sv.optim.CubicSpline(xs, ys)
        # At knot points, should be exact
        assert abs(spline.eval(0.0) - 0.0) < 1e-10
        assert abs(spline.eval(1.0) - 1.0) < 1e-10
        assert abs(spline.eval(2.0) - 4.0) < 1e-10

    def test_eval_many(self):
        xs = [0.0, 1.0, 2.0, 3.0]
        ys = [0.0, 1.0, 4.0, 9.0]
        spline = sv.optim.CubicSpline(xs, ys)
        result = spline.eval_many([0.5, 1.5, 2.5])
        assert len(result) == 3


# ===========================================================================
# MINIMIZATION
# ===========================================================================


class TestMinimize:
    def test_nelder_mead_quadratic(self):
        """Minimize (x-3)^2 → minimum at x=3."""
        r = sv.optim.minimize(
            lambda x: (x.tolist()[0] - 3.0) ** 2,
            sv.Tensor([0.0], [1]),
            method="nelder-mead",
        )
        assert r["converged"]
        x = r["x"].tolist()
        assert abs(x[0] - 3.0) < 0.01

    def test_bfgs_rosenbrock(self):
        """Minimize Rosenbrock: f(x,y) = (1-x)^2 + 100*(y-x^2)^2."""
        def rosenbrock(t):
            d = t.tolist()
            x, y = d[0], d[1]
            return (1 - x) ** 2 + 100 * (y - x**2) ** 2

        r = sv.optim.minimize(
            rosenbrock,
            sv.Tensor([0.0, 0.0], [2]),
            method="bfgs",
            max_iter=5000,
            gtol=1e-6,
        )
        x = r["x"].tolist()
        # Minimum at (1, 1)
        assert abs(x[0] - 1.0) < 0.1
        assert abs(x[1] - 1.0) < 0.1

    def test_gradient_descent(self):
        """Minimize x^2 + y^2 with explicit gradient."""
        r = sv.optim.minimize(
            lambda t: sum(v**2 for v in t.tolist()),
            sv.Tensor([5.0, 3.0], [2]),
            method="gradient-descent",
            jac=lambda t: sv.Tensor([2 * v for v in t.tolist()], [2]),
            learning_rate=0.1,
            max_iter=1000,
        )
        x = r["x"].tolist()
        assert abs(x[0]) < 0.1
        assert abs(x[1]) < 0.1


# ===========================================================================
# 1-D MINIMIZATION
# ===========================================================================


class TestGoldenSection:
    def test_quadratic(self):
        """Minimize (x-2)^2 on [0, 5]."""
        r = sv.optim.golden_section(lambda x: (x - 2) ** 2, 0.0, 5.0)
        assert r["converged"]
        assert abs(r["x_min"] - 2.0) < 1e-8

    def test_sin(self):
        """Find min of -sin(x) on [0, π] → min at π/2."""
        r = sv.optim.golden_section(lambda x: -math.sin(x), 0.0, math.pi)
        assert abs(r["x_min"] - math.pi / 2) < 1e-6


class TestBrentMin:
    def test_quadratic(self):
        r = sv.optim.brent_min(lambda x: (x - 3) ** 2, 0.0, 6.0)
        assert r["converged"]
        assert abs(r["x_min"] - 3.0) < 1e-8


# ===========================================================================
# ODE SOLVERS
# ===========================================================================


class TestSolveIVP:
    def test_exponential_decay(self):
        """dy/dt = -y, y(0) = 1 → y(t) = e^(-t)."""
        r = sv.optim.solve_ivp(
            lambda t, y: [-y[0]],
            [0.0, 2.0],
            [1.0],
            method="rk45",
        )
        assert r["success"]
        # At t=2, y ≈ e^(-2) ≈ 0.1353
        t_vals = r["t"]
        y_vals = r["y"]
        assert len(t_vals) > 2
        final_y = y_vals[-1][0]
        assert abs(final_y - math.exp(-2.0)) < 0.01

    def test_euler(self):
        """Simple test with Euler method."""
        r = sv.optim.solve_ivp(
            lambda t, y: [1.0],  # dy/dt = 1 → y = t + 1
            [0.0, 1.0],
            [1.0],
            method="euler",
        )
        assert r["success"]
        final_y = r["y"][-1][0]
        assert abs(final_y - 2.0) < 0.1


# ===========================================================================
# CURVE FITTING
# ===========================================================================


class TestCurveFit:
    def test_linear_fit(self):
        """Fit y = a*x + b to data."""
        x_data = [0.0, 1.0, 2.0, 3.0, 4.0]
        y_data = [1.0, 3.0, 5.0, 7.0, 9.0]  # y = 2x + 1
        r = sv.optim.curve_fit(
            lambda x, p: p[0] * x + p[1],
            x_data,
            y_data,
            p0=[1.0, 0.0],
        )
        assert r["converged"]
        params = r["params"]
        assert abs(params[0] - 2.0) < 0.1
        assert abs(params[1] - 1.0) < 0.1

    def test_exponential_fit(self):
        """Fit y = a * exp(b * x)."""
        x_data = [0.0, 0.5, 1.0, 1.5, 2.0]
        y_data = [1.0 * math.exp(0.5 * x) for x in x_data]
        r = sv.optim.curve_fit(
            lambda x, p: p[0] * math.exp(p[1] * x),
            x_data,
            y_data,
            p0=[1.0, 1.0],
        )
        assert r["converged"]
        assert abs(r["params"][0] - 1.0) < 0.2
        assert abs(r["params"][1] - 0.5) < 0.2


# ===========================================================================
# LINEAR PROGRAMMING
# ===========================================================================


class TestLinProg:
    def test_simple_lp(self):
        """Minimize -x - 2y subject to x + y <= 4, x <= 3, y <= 3, x,y >= 0."""
        r = sv.optim.linprog(
            c=[-1.0, -2.0],
            a_ub=[[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
            b_ub=[4.0, 3.0, 3.0],
        )
        assert r["converged"]
        x = r["x"]
        # Optimal: x=1, y=3 → fun = -7
        assert abs(r["fun"] - (-7.0)) < 0.1


# ===========================================================================
# PDE SOLVERS
# ===========================================================================


class TestHeatEquation:
    def test_basic(self):
        """Heat equation with initial sin(πx) on [0,1]."""
        r = sv.optim.heat_equation_1d(
            initial=lambda x: math.sin(math.pi * x),
            x_range=(0.0, 1.0),
            n_x=20,
            t_final=0.1,
            n_t=100,
            alpha=1.0,
        )
        assert "u" in r
        assert "x" in r
        assert "t" in r
        assert len(r["x"]) == 20


# ===========================================================================
# INTEGRATION TESTS
# ===========================================================================


class TestIntegration:
    def test_all_functions_accessible(self):
        """All optim submodule functions should be importable."""
        fns = [
            sv.optim.bisection,
            sv.optim.brentq,
            sv.optim.newton,
            sv.optim.trapezoid,
            sv.optim.simpson,
            sv.optim.quad,
            sv.optim.interp1d,
            sv.optim.CubicSpline,
            sv.optim.minimize,
            sv.optim.golden_section,
            sv.optim.brent_min,
            sv.optim.solve_ivp,
            sv.optim.curve_fit,
            sv.optim.linprog,
            sv.optim.heat_equation_1d,
        ]
        for fn in fns:
            assert fn is not None
