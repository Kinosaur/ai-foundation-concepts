import sympy as sp

# Define the variable and function
x = sp.symbols('x')
y = 2*x**4 - 7*x**3

# 1. Calculate the Derivative
y_prime = sp.diff(y, x)
print(f"Derivative: {y_prime}")

# 2. Solve for where Derivative equals 0
critical_points = sp.solve(y_prime, x)
print(f"Critical Points (Slope is 0): {critical_points}")