# Matrix Multiplication

**Difficulty:** Easy  
**Category:** Linear Algebra  
**Tags:** [matrices, multiplication, fundamental]

## Problem Description

Implement matrix multiplication from scratch. Given two matrices A (m×n) and B (n×p), compute their product C = A × B, where C is an (m×p) matrix.

## Mathematical Formulation

For matrices A (m×n) and B (n×p), the product C = AB is defined as:

$$
C_{ij} = \sum_{k=1}^{n} A_{ik} \cdot B_{kj}
$$

where $C_{ij}$ is the element in the i-th row and j-th column of the result matrix.

## Constraints

- The number of columns in matrix A must equal the number of rows in matrix B
- All elements are real numbers
- Matrices are represented as 2D lists/arrays

## Examples

### Example 1
```
Input: 
A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]

Output: 
[[19, 22], [43, 50]]

Explanation:
C[0][0] = 1*5 + 2*7 = 19
C[0][1] = 1*6 + 2*8 = 22
C[1][0] = 3*5 + 4*7 = 43
C[1][1] = 3*6 + 4*8 = 50
```

### Example 2
```
Input: 
A = [[1, 2, 3]]
B = [[4], [5], [6]]

Output: 
[[32]]

Explanation:
C[0][0] = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
```

## Solution Approach

1. Check if matrices can be multiplied (columns of A = rows of B)
2. Initialize result matrix with zeros
3. For each element in result matrix, compute dot product of corresponding row from A and column from B

## Implementation

### Python
```python
def matrix_multiply(A, B):
    """
    Multiply two matrices A and B.
    
    Args:
        A: List[List[float]] - First matrix (m x n)
        B: List[List[float]] - Second matrix (n x p)
    
    Returns:
        List[List[float]] - Result matrix (m x p)
    """
    # Get dimensions
    m, n = len(A), len(A[0])
    n2, p = len(B), len(B[0])
    
    # Check if multiplication is possible
    if n != n2:
        raise ValueError("Matrix dimensions incompatible for multiplication")
    
    # Initialize result matrix with zeros
    C = [[0 for _ in range(p)] for _ in range(m)]
    
    # Compute matrix multiplication
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    
    return C
```

### JavaScript
```javascript
function matrixMultiply(A, B) {
    // Get dimensions
    const m = A.length;
    const n = A[0].length;
    const n2 = B.length;
    const p = B[0].length;
    
    // Check if multiplication is possible
    if (n !== n2) {
        throw new Error("Matrix dimensions incompatible for multiplication");
    }
    
    // Initialize result matrix with zeros
    const C = Array(m).fill(0).map(() => Array(p).fill(0));
    
    // Compute matrix multiplication
    for (let i = 0; i < m; i++) {
        for (let j = 0; j < p; j++) {
            for (let k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    
    return C;
}
```

## Test Cases

```python
# Test case 1: 2x2 matrices
A1 = [[1, 2], [3, 4]]
B1 = [[5, 6], [7, 8]]
assert matrix_multiply(A1, B1) == [[19, 22], [43, 50]]

# Test case 2: 1x3 and 3x1 matrices
A2 = [[1, 2, 3]]
B2 = [[4], [5], [6]]
assert matrix_multiply(A2, B2) == [[32]]

# Test case 3: 3x2 and 2x3 matrices
A3 = [[1, 2], [3, 4], [5, 6]]
B3 = [[1, 2, 3], [4, 5, 6]]
expected = [[9, 12, 15], [19, 26, 33], [29, 40, 51]]
assert matrix_multiply(A3, B3) == expected

# Test case 4: Identity matrix
A4 = [[1, 0], [0, 1]]
B4 = [[5, 6], [7, 8]]
assert matrix_multiply(A4, B4) == B4
```

## Time Complexity

**O(m × n × p)** where:
- m = number of rows in matrix A
- n = number of columns in A (and rows in B)
- p = number of columns in matrix B

## Space Complexity

**O(m × p)** for storing the result matrix.

## References

- [Matrix Multiplication - Wikipedia](https://en.wikipedia.org/wiki/Matrix_multiplication)
- Gilbert Strang, "Introduction to Linear Algebra"
