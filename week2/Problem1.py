import numpy as np

# Define Matrix A (Coefficients) and Matrix B (Results)
A = np.array([
    [1, 1, 1],
    [1, 2, 1],
    [1, 1, 2]
])

B = np.array([
    [10],
    [15],
    [12]
])

# 1. Check if matrix is singular (Determinant check)
det_A = np.linalg.det(A)
print(f"Determinant of A: {det_A}")

if det_A != 0:
    print("Matrix is non-singular and invertible.")
    
    # 2. Calculate Inverse
    A_inv = np.linalg.inv(A)
    print("\nInverse of Matrix A:")
    print(A_inv)
    
    # 3. Solve for X (Prices)
    # Formula: X = A_inv * B
    X = np.dot(A_inv, B)
    print("\nSolution (Prices):")
    print(f"Apple (x): ${X[0][0]:.2f}")
    print(f"Banana (y): ${X[1][0]:.2f}")
    print(f"Cherry (z): ${X[2][0]:.2f}")

else:
    print("Matrix is singular and cannot be inverted.")