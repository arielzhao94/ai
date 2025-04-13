# 1.Linear Algebra

## Matrix-Vector Dot Product
```
def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	# Return a list where each element is the dot product of a row of 'a' with 'b'.
	# If the number of columns in 'a' does not match the length of 'b', return -1.
	if len(a[0]) != len(b):
        return -1
    else:
        res = []
        for l in a:
            sum = 0
            for i in range(len(l)):
                sum += l[i] * b[i]
            res.append(sum)
        return res
```


## Transpose of a Matrix
```
def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    b = []
    for i in range(len(a[0])):
        temp = []
        for j in range(len(a)):
            
            temp.append(a[j][i])
        b.append(temp)
	return b
```


*Solution*
```
def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:

    return [list(i) for i in zip(*a)]
```

## Dot Product Calculator
```
import numpy as np

def calculate_dot_product(vec1, vec2) -> float:
	"""
	Calculate the dot product of two vectors.
	Args:
		vec1 (numpy.ndarray): 1D array representing the first vector.
		vec2 (numpy.ndarray): 1D array representing the second vector.
	"""
	# Your code here
    total = 0
	for i in range(len(vec1)):
        total += vec1[i] * vec2[i]
    return total
```

*Solution*
```import numpy as np

def calculate_dot_product(vec1, vec2):
    """
    Calculate the dot product of two vectors.
    Args:
        vec1 (numpy.ndarray): 1D array representing the first vector.
        vec2 (numpy.ndarray): 1D array representing the second vector.
    Returns:
        float: Dot product of the two vectors.
    """
    return np.dot(vec1, vec2)
```

## Scalar Multiplication of a Matrix
```
def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
    result = []
    for i in matrix:
        temp = []
        for j in i:
            j *= scalar
            temp.append(j)
        result.append(temp)
	return result
```
*Solution*
```
def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
    return [[element * scalar for element in row] for row in matrix]
```
## Calculate Cosine Similarity Between Vectors
```
import numpy as np

def cosine_similarity(v1, v2):
	# Implement your code here
	if len(v1) == 0 or len(v2) == 0:
        return -1

    if len(v1) != len(v2):
        return -1

    p = 0
    m1 = 0
    m2 = 0
    for i in range(len(v1)):
        p += v1[i] * v2[i]
        m1 += v1[i]**2  
        m2 += v2[i]**2
    return f'{p/((m1 * m2)**0.5):.3f}'
```
*Solution*
```
import numpy as np

def cosine_similarity(v1, v2):
    if v1.shape != v2.shape:
        raise ValueError("Arrays must have the same shape")

    if v1.size == 0:
        raise ValueError("Arrays cannot be empty")

    # Flatten arrays in case of 2D
    v1_flat = v1.flatten()
    v2_flat = v2.flatten()

    dot_product = np.dot(v1_flat, v2_flat)
    magnitude1 = np.sqrt(np.sum(v1_flat**2))
    magnitude2 = np.sqrt(np.sum(v2_flat**2))
    # YX: a non-empty vector can still have all-zero elements, [0, 0, 0]
    if magnitude1 == 0 or magnitude2 == 0:
        raise ValueError("Vectors cannot have zero magnitude")

    return round(dot_product / (magnitude1 * magnitude2), 3)
```
## Calculate Mean by Row or Column
```
def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    

    means = []
    if mode.lower() == 'row':
        for i in matrix:
            total = sum(i)
            means.append(total/len(i))

    if mode.lower() == 'column':
        for i in range(len(matrix[0])):
            total = 0
            for j in range(len(matrix)):
                total += matrix[j][i] 
            means.append(total/len(matrix))

	return means
```
*Solution*
```
def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    if mode == 'column':
        return [sum(col) / len(matrix) for col in zip(*matrix)]
    elif mode == 'row':
        return [sum(row) / len(row) for row in matrix]
    else:
        raise ValueError("Mode must be 'row' or 'column'")
```

## Calculate Eigenvalues of a Matrix
*Solution*
```
def calculate_eigenvalues(matrix: list[list[float]]) -> list[float]:
    a, b, c, d = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
    trace = a + d
    determinant = a * d - b * c
    # Calculate the discriminant of the quadratic equation
    discriminant = trace**2 - 4 * determinant
    # Solve for eigenvalues
    lambda_1 = (trace + discriminant**0.5) / 2
    lambda_2 = (trace - discriminant**0.5) / 2
    return [lambda_1, lambda_2]
```
## Calculate 2x2 Matrix Inverse
```
def inverse_2x2(matrix: list[list[float]]) -> list[list[float]]:
    a,b,c,d = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
    if a*d - b*c == 0:
        return None
    else:
        determinant = 1/(a*d - b*c)
        inverse = [[d*determinant, -b*determinant], [-c*determinant, a*determinant]]

        return inverse
```
*Solution*
```
def inverse_2x2(matrix: list[list[float]]) -> list[list[float]]:
    a, b, c, d = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
    determinant = a * d - b * c
    if determinant == 0:
        return None
    inverse = [[d/determinant, -b/determinant], [-c/determinant, a/determinant]]
    return inverse
```
## Matrix times Matrix
