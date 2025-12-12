import numpy as np

singular_matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [1, 2, 3]  # Repeating the first row makes this matrix singular
])

# Calculate the determinant of the singular matrix
det_singular_matrix = np.linalg.det(singular_matrix)

# Try to calculate the inverse, which should raise an error since det = 0
try:
    inverse_singular_matrix = np.linalg.inv(singular_matrix)
except np.linalg.LinAlgError as e:
    inverse_singular_matrix = str(e)

print("det_singular_matrix:", det_singular_matrix)
print("inverse_singular_matrix:", inverse_singular_matrix)