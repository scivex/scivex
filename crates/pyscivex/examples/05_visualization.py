"""Visualization — line, scatter, histogram, save to SVG."""
import pyscivex as sv

# Line plot
fig = sv.Figure()
x = [1.0, 2.0, 3.0, 4.0, 5.0]
y = [1.0, 4.0, 9.0, 16.0, 25.0]
fig.line_plot(x, y, label="y = x^2")
fig.title("Quadratic Function")
fig.xlabel("x")
fig.ylabel("y")
fig.grid(True)
fig.save_svg("line_plot.svg")
print("Saved line_plot.svg")

# Scatter plot
fig2 = sv.Figure()
fig2.scatter([1, 2, 3, 4, 5], [2.1, 3.9, 6.2, 7.8, 10.1], label="data")
fig2.title("Scatter Example")
fig2.save_svg("scatter_plot.svg")
print("Saved scatter_plot.svg")

# Histogram
fig3 = sv.Figure()
import random
data = [random.gauss(0, 1) for _ in range(1000)]
fig3.histogram(data, bins=30)
fig3.title("Normal Distribution")
fig3.save_svg("histogram.svg")
print("Saved histogram.svg")

# Generate matplotlib script
script = fig.to_matplotlib_script()
print("\nMatplotlib equivalent:")
print(script)
