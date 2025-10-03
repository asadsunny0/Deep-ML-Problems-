'''
	let , a is 2D Matrix
          b is 1D (Vector)

    We need to perform  a x b (a's column and b's row should be same)
'''
def matrix_dot(a:list[list[int|float]], b:list[int|float]) -> list[int|float] :
    if len(a[0]) != len(b):
        return -1
    result = []
    for row in a:
        total = 0
        for i in range(len(row)):
            total += row[i] * b[i]
        result.append(total)
    return result

# ----------------------
# Example usage
# ----------------------

# Example 1: Valid case
a = [[1, 2, 3],
    [4, 5, 6]]
b = [7, 8, 9]

print("Matrix:")
for row in a:
    print(row)
print("Vector:", b)

result = matrix_dot(a, b)
print("Result:", result)  # Expected: [50, 122]

# Example 2: Invalid case
a2 = [[1, 2],
      [3, 4]]
b2 = [5, 6, 7]

print("\nMatrix:")
for row in a2:
    print(row)
print("Vector:", b2)

result2 = matrix_dot(a2, b2)
print("Result:", result2)  # Expected: -1 (dimension mismatch)