import numpy as np

A = np.array([[3, 0, 2], [2, 0, -2], [0, 1, 1]])

det_A = np.linalg.det(A)

if det_A != 0:
    A_inv = np.linalg.inv(A)
else:
    A_inv = None
print("=" * 20)
print("det_A:", det_A)
print("=" * 20)
print("A_inv:")
print(A_inv)
print("=" * 20)
print("AxA_inv:")
print(np.matmul(A, A_inv))
print("=" * 20)

identity_check_rounded = np.round(np.matmul(A, A_inv), 8)
print(identity_check_rounded)
result_check = np.allclose(identity_check_rounded, np.eye(3, 3))
print(result_check)
