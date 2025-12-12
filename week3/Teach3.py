import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

# Define training set
X = np.array([-3, -1, 0.0, 1, 3]).reshape(-1,1) # 5x1 vector, N=5, D=1
y = np.array([2.2, 3.7, 3.14, 3.67, 4.67]).reshape(-1,1) # 5x1 vector

# Plot the training set
plt.figure()
plt.plot(X, y, '+', markersize=10, label='Data points')
plt.xlabel("X-axis ($x$)")
plt.ylabel("Y-axis ($y$)")
plt.title("Plot of Training Data Set")
plt.xlim([-5, 5]) # Setting x-axis limits
plt.legend()
plt.grid(True)
plt.show()

