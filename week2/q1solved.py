import numpy as np
import matplotlib.pyplot as plt

# x range
x = np.linspace(-2, 4, 1000)

# function and derivatives
y = 2*x**4 - 7*x**3
y_prime = 8*x**3 - 21*x**2
y_double_prime = 24*x**2 - 42*x

# plotting
plt.figure(figsize=(12, 8))

# original function
plt.subplot(3, 1, 1)
plt.plot(x, y, label=r'$y = 2x^4 - 7x^3$')
plt.title('Function and Derivatives')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()

# first derivative
plt.subplot(3, 1, 2)
plt.plot(x, y_prime, 'orange', label=r"$y' = 8x^3 - 21x^2$")
plt.axhline(0, color='k', linewidth=0.5)
plt.axvline(0, color='k', linewidth=0.5)
plt.scatter([0, 21/8], [0, 0], c='red')  # critical pts
plt.xlabel('x')
plt.ylabel(r"$y'$")
plt.grid(True)
plt.legend()

# second derivative
plt.subplot(3, 1, 3)
plt.plot(x, y_double_prime, 'g', label=r"$y'' = 24x^2 - 42x$")
plt.axhline(0, color='k', linewidth=0.5)
plt.axvline(0, color='k', linewidth=0.5)
plt.scatter([21/8], [55.125], c='blue')  # inflection pt
plt.xlabel('x')
plt.ylabel(r"$y''$")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# check value at critical point
x_crit = 21/8
y_val = 2*x_crit**4 - 7*x_crit**3
print("y(21/8):", y_val)