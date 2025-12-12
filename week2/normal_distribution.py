import numpy as np
import matplotlib.pyplot as plt

# Normal distribution function
def normal_dist(x, mu, sigma):
    """Calculate the Gaussian/normal distribution."""
    coefficient = 1 / (sigma * np.sqrt(2 * np.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coefficient * np.exp(exponent)

# Set parameters
mu = 5      # mean
sigma = 1   # standard deviation

# Create x values range
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
print("X values:", x[:5], "...", x[-5:])

# Calculate y values
y = normal_dist(x, mu, sigma)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(x, y, 'b-', label=f"Normal Distribution (μ={mu}, σ={sigma})")
plt.title("Normal Distribution Curve")
plt.xlabel("x")
plt.ylabel("Probability Density")
plt.legend()
plt.grid(True)
plt.show()
