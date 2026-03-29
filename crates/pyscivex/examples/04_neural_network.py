"""Neural network training — Sequential model with Adam optimizer."""
import pyscivex as sv

# Build a simple model: 2 -> 16 -> 1
model = sv.nn.Sequential([
    sv.nn.Linear(2, 16, seed=42),
    sv.nn.ReLU(),
    sv.nn.Linear(16, 1, seed=43),
])

# Optimizer
optimizer = sv.nn.Adam(model.parameters(), lr=0.01)

# Simple training data: y = x1 + x2
x_data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
y_data = [[3.0], [7.0], [11.0], [15.0]]

# Training loop
for epoch in range(100):
    total_loss = 0.0
    for x_row, y_row in zip(x_data, y_data):
        x = sv.nn.tensor([x_row], requires_grad=True)
        y = sv.nn.tensor([y_row])

        pred = model.forward(x)
        loss = sv.nn.mse_loss(pred, y)
        loss.backward()

        total_loss += loss.data().to_list()[0]

    optimizer.step()
    optimizer.zero_grad()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}: loss = {total_loss / len(x_data):.4f}")

# Test
test_x = sv.nn.tensor([[2.0, 3.0]], requires_grad=False)
test_pred = model.forward(test_x)
print(f"\nPrediction for [2, 3]: {test_pred.data().to_list()} (expected ~5.0)")
