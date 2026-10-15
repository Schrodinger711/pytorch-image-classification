import torch

x = torch.tensor(3.0)
w = torch.tensor(2.8, requires_grad=True)
actual = torch.tensor(6.0)

prediction = x * w


print(f"Prediction: {prediction}")

loss = (prediction - actual) ** 2
print(f"Loss: {loss}")

loss.backward()
print(f"Gradient of w: {w.grad}")

learning_rate = 0.01
with torch.no_grad():
    w -= learning_rate * w.grad 
print(f"Updated w: {w}")   
