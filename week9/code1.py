import numpy as np

# Generate data: y = 2x + 1
X = np.linspace(-5, 5, 50).reshape(-1, 1)
T = 2*X + 1

# Initialize parameters
w = np.random.randn()
b = np.random.randn()
# Hyperparameters

eta = 0.1
epochs = 50

for epoch in range(epochs):
    # Forward pass
    y = w*X + b
    
    # Compute error
    E = np.mean((T - y)**2)
    
    # Gradients
    dE_dw = np.mean(2*X*(y - T))
    dE_db = np.mean(2*(y - T))
    
    # Update
    w -= eta*dE_dw
    b -= eta*dE_db
    
    if (epoch+1) % 1 == 0:
        print(f"Epoch {epoch+1}: Error={E:.4f}")
        print(f"Weight epoch {epoch+1}: {w}")
        print(f"bias epoch {epoch+1}: {b}")

print(f"Trained w: {w:.2f}, b: {b:.2f}")
