"""Tests for pyscivex visualization — Figure and viz submodule."""

import pyscivex as sv


# ===========================================================================
# FIGURE BASICS
# ===========================================================================


class TestFigureBasics:
    def test_create(self):
        fig = sv.Figure()
        assert repr(fig)

    def test_create_with_size(self):
        fig = sv.Figure(width=1200.0, height=800.0)
        assert "1200" in repr(fig)

    def test_title_labels(self):
        fig = sv.Figure()
        fig.title("Test Plot")
        fig.x_label("X Axis")
        fig.y_label("Y Axis")
        fig.line_plot([1.0, 2.0], [3.0, 4.0])
        svg = fig.to_svg()
        assert "Test Plot" in svg

    def test_save_svg(self, tmp_path):
        fig = sv.Figure()
        fig.line_plot([1.0, 2.0], [3.0, 4.0])
        path = str(tmp_path / "test.svg")
        fig.save_svg(path)
        with open(path) as f:
            content = f.read()
        assert "<svg" in content

    def test_repr_svg(self):
        fig = sv.Figure()
        fig.line_plot([1.0, 2.0], [3.0, 4.0])
        svg = fig._repr_svg_()
        assert "<svg" in svg

    def test_grid(self):
        fig = sv.Figure()
        fig.grid(True)
        fig.line_plot([1.0, 2.0], [3.0, 4.0])
        svg = fig.to_svg()
        assert isinstance(svg, str)

    def test_theme(self):
        fig = sv.Figure()
        fig.theme("dark")
        fig.line_plot([1.0, 2.0], [3.0, 4.0])
        svg = fig.to_svg()
        assert isinstance(svg, str)

    def test_to_terminal(self):
        fig = sv.Figure()
        fig.line_plot([1.0, 2.0, 3.0], [1.0, 4.0, 2.0])
        term = fig.to_terminal()
        assert isinstance(term, str)
        assert len(term) > 0


# ===========================================================================
# BASIC PLOT TYPES
# ===========================================================================


class TestLinePlot:
    def test_basic(self):
        fig = sv.Figure()
        fig.line_plot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        svg = fig.to_svg()
        assert "<svg" in svg

    def test_with_options(self):
        fig = sv.Figure()
        fig.line_plot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0],
                      label="data", color="#ff0000", width=2.0)
        svg = fig.to_svg()
        assert isinstance(svg, str)


class TestScatterPlot:
    def test_basic(self):
        fig = sv.Figure()
        fig.scatter_plot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        svg = fig.to_svg()
        assert "<svg" in svg

    def test_with_options(self):
        fig = sv.Figure()
        fig.scatter_plot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0],
                         label="points", color="#00ff00", size=5.0)
        svg = fig.to_svg()
        assert isinstance(svg, str)


class TestBarPlot:
    def test_basic(self):
        fig = sv.Figure()
        fig.bar_plot(["A", "B", "C"], [10.0, 20.0, 15.0])
        svg = fig.to_svg()
        assert "<svg" in svg

    def test_with_label(self):
        fig = sv.Figure()
        fig.bar_plot(["X", "Y", "Z"], [5.0, 10.0, 7.0], label="sales", color="#0000ff")
        svg = fig.to_svg()
        assert isinstance(svg, str)


class TestHistogram:
    def test_basic(self):
        import math
        data = [math.sin(i * 0.1) for i in range(100)]
        fig = sv.Figure()
        fig.histogram(data, bins=15)
        svg = fig.to_svg()
        assert "<svg" in svg

    def test_with_options(self):
        data = [float(i) for i in range(50)]
        fig = sv.Figure()
        fig.histogram(data, bins=10, label="dist", color="#ff8800")
        svg = fig.to_svg()
        assert isinstance(svg, str)


# ===========================================================================
# STATISTICAL PLOT TYPES
# ===========================================================================


class TestBoxPlot:
    def test_basic(self):
        datasets = [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 3.0, 4.0, 5.0, 6.0],
            [0.5, 1.5, 2.5, 3.5, 4.5],
        ]
        fig = sv.Figure()
        fig.boxplot(datasets, labels=["A", "B", "C"])
        svg = fig.to_svg()
        assert "<svg" in svg


class TestViolinPlot:
    def test_basic(self):
        datasets = [
            [1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0],
            [2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0],
        ]
        fig = sv.Figure()
        fig.violin_plot(datasets)
        svg = fig.to_svg()
        assert "<svg" in svg


# ===========================================================================
# HEATMAP & PIE
# ===========================================================================


class TestHeatmap:
    def test_basic(self):
        data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        fig = sv.Figure()
        fig.heatmap(data)
        svg = fig.to_svg()
        assert "<svg" in svg

    def test_with_labels(self):
        data = [[1.0, 0.5], [0.5, 1.0]]
        fig = sv.Figure()
        fig.heatmap(data, x_labels=["A", "B"], y_labels=["X", "Y"], show_values=True)
        svg = fig.to_svg()
        assert isinstance(svg, str)


class TestPieChart:
    def test_basic(self):
        fig = sv.Figure()
        fig.pie_chart([30.0, 40.0, 30.0], labels=["A", "B", "C"])
        svg = fig.to_svg()
        assert "<svg" in svg


# ===========================================================================
# SPECIALTY PLOT TYPES
# ===========================================================================


class TestAreaPlot:
    def test_basic(self):
        fig = sv.Figure()
        fig.area_plot([1.0, 2.0, 3.0, 4.0], [1.0, 3.0, 2.0, 4.0], label="area")
        svg = fig.to_svg()
        assert "<svg" in svg


class TestErrorBarPlot:
    def test_basic(self):
        fig = sv.Figure()
        fig.error_bar_plot(
            [1.0, 2.0, 3.0],
            [5.0, 6.0, 7.0],
            [0.5, 0.3, 0.4],
            [0.5, 0.3, 0.4],
        )
        svg = fig.to_svg()
        assert "<svg" in svg


class TestContourPlot:
    def test_basic(self):
        data = [[float(i + j) for j in range(10)] for i in range(10)]
        fig = sv.Figure()
        fig.contour_plot(data, n_levels=5)
        svg = fig.to_svg()
        assert "<svg" in svg


class TestPolarPlot:
    def test_basic(self):
        fig = sv.Figure()
        fig.polar_plot(["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
                       [4.0, 3.0, 5.0, 2.0, 3.0, 4.0, 6.0, 3.5])
        svg = fig.to_svg()
        assert "<svg" in svg


class TestConfidenceBand:
    def test_basic(self):
        fig = sv.Figure()
        x = [1.0, 2.0, 3.0, 4.0]
        fig.confidence_band(x, [0.5, 1.5, 2.5, 3.5], [1.5, 2.5, 3.5, 4.5])
        svg = fig.to_svg()
        assert "<svg" in svg


# ===========================================================================
# STATISTICAL VIZ
# ===========================================================================


class TestRegressionPlot:
    def test_basic(self):
        fig = sv.Figure()
        fig.regression_plot([1.0, 2.0, 3.0, 4.0, 5.0],
                           [2.1, 3.9, 6.2, 7.8, 10.1])
        svg = fig.to_svg()
        assert "<svg" in svg


class TestResidualPlot:
    def test_basic(self):
        fig = sv.Figure()
        fig.residual_plot([1.0, 2.0, 3.0, 4.0], [2.1, 3.9, 6.2, 7.8])
        svg = fig.to_svg()
        assert "<svg" in svg


class TestQQPlot:
    def test_basic(self):
        import math
        data = [math.sin(i * 0.1) * 3 + i * 0.5 for i in range(30)]
        fig = sv.Figure()
        fig.qq_plot(data)
        svg = fig.to_svg()
        assert "<svg" in svg


class TestCorrelationHeatmap:
    def test_basic(self):
        matrix = [[1.0, 0.8, 0.3], [0.8, 1.0, 0.5], [0.3, 0.5, 1.0]]
        fig = sv.Figure()
        fig.correlation_heatmap(matrix, labels=["A", "B", "C"])
        svg = fig.to_svg()
        assert "<svg" in svg


# ===========================================================================
# VIZ SUBMODULE CONVENIENCE FUNCTIONS
# ===========================================================================


class TestVizSubmodule:
    def test_regplot(self):
        fig = sv.viz.regplot([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
        svg = fig.to_svg()
        assert "<svg" in svg

    def test_residplot(self):
        fig = sv.viz.residplot([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
        svg = fig.to_svg()
        assert "<svg" in svg

    def test_qqplot(self):
        data = [float(i) for i in range(20)]
        fig = sv.viz.qqplot(data)
        svg = fig.to_svg()
        assert "<svg" in svg

    def test_corrplot(self):
        matrix = [[1.0, 0.5], [0.5, 1.0]]
        fig = sv.viz.corrplot(matrix, labels=["X", "Y"])
        svg = fig.to_svg()
        assert "<svg" in svg


# ===========================================================================
# MULTIPLE PLOTS ON ONE FIGURE
# ===========================================================================


class TestMultiplePlots:
    def test_line_and_scatter(self):
        fig = sv.Figure()
        fig.line_plot([1.0, 2.0, 3.0], [1.0, 4.0, 9.0], label="trend")
        fig.scatter_plot([1.0, 2.0, 3.0], [1.2, 3.8, 9.3], label="data")
        svg = fig.to_svg()
        assert "<svg" in svg

    def test_line_with_confidence_band(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        fig = sv.Figure()
        fig.line_plot(x, [2.0, 4.0, 6.0, 8.0, 10.0], label="mean")
        fig.confidence_band(x, [1.5, 3.5, 5.5, 7.5, 9.5],
                              [2.5, 4.5, 6.5, 8.5, 10.5], label="95% CI")
        svg = fig.to_svg()
        assert "<svg" in svg


# ===========================================================================
# INTEGRATION
# ===========================================================================


class TestIntegration:
    def test_all_accessible(self):
        """All viz submodule items should be importable."""
        assert sv.Figure is not None
        assert sv.viz.Figure is not None
        assert sv.viz.regplot is not None
        assert sv.viz.residplot is not None
        assert sv.viz.qqplot is not None
        assert sv.viz.corrplot is not None
