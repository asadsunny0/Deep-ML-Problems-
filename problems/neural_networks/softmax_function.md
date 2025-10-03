# Softmax Function Implementation

**Difficulty:** Easy  
**Category:** Neural Networks  
**Tags:** [activation-function, softmax, classification]

## Problem Description

Implement the softmax activation function from scratch. The softmax function converts a vector of real numbers into a probability distribution, where all values are in the range (0, 1) and sum to 1.

## Mathematical Formulation

For a vector $\mathbf{x} = [x_1, x_2, ..., x_n]$, the softmax function is defined as:

$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}
$$

To improve numerical stability, we use the following equivalent formulation:

$$
\text{softmax}(x_i) = \frac{e^{x_i - \max(\mathbf{x})}}{\sum_{j=1}^{n} e^{x_j - \max(\mathbf{x})}}
$$

## Constraints

- Input is a 1D array/list of real numbers
- Output should sum to 1.0 (within numerical precision)
- All output values should be positive

## Examples

### Example 1
```
Input: [1.0, 2.0, 3.0]
Output: [0.09003057, 0.24472847, 0.66524096]

Explanation:
e^1 = 2.718, e^2 = 7.389, e^3 = 20.086
sum = 30.193
softmax = [2.718/30.193, 7.389/30.193, 20.086/30.193]
        = [0.09003057, 0.24472847, 0.66524096]
```

### Example 2
```
Input: [1.0, 1.0, 1.0]
Output: [0.33333333, 0.33333333, 0.33333333]

Explanation:
All inputs are equal, so the probability is uniformly distributed.
```

### Example 3
```
Input: [0.0, 0.0, 0.0]
Output: [0.33333333, 0.33333333, 0.33333333]

Explanation:
e^0 = 1 for all elements, resulting in uniform distribution.
```

## Solution Approach

1. Find the maximum value in the input array (for numerical stability)
2. Subtract the maximum from each element
3. Compute exponential of each element
4. Divide each exponential by the sum of all exponentials

The subtraction of the maximum value prevents overflow when computing exponentials of large numbers.

## Implementation

### Python
```python
import math

def softmax(x):
    """
    Compute the softmax of a vector x.
    
    Args:
        x: List[float] - Input vector
    
    Returns:
        List[float] - Softmax probabilities
    """
    # Numerical stability: subtract max value
    max_x = max(x)
    exp_x = [math.exp(xi - max_x) for xi in x]
    
    # Compute sum of exponentials
    sum_exp_x = sum(exp_x)
    
    # Normalize to get probabilities
    return [exp_xi / sum_exp_x for exp_xi in exp_x]

# NumPy version (more efficient for large arrays)
def softmax_numpy(x):
    """
    Compute softmax using NumPy.
    
    Args:
        x: np.ndarray - Input vector
    
    Returns:
        np.ndarray - Softmax probabilities
    """
    import numpy as np
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)
```

### JavaScript
```javascript
function softmax(x) {
    // Find maximum value for numerical stability
    const maxX = Math.max(...x);
    
    // Compute exponentials with stability adjustment
    const expX = x.map(xi => Math.exp(xi - maxX));
    
    // Compute sum of exponentials
    const sumExpX = expX.reduce((a, b) => a + b, 0);
    
    // Normalize to get probabilities
    return expX.map(exp_xi => exp_xi / sumExpX);
}
```

## Test Cases

```python
import math

# Test case 1: Basic test
result1 = softmax([1.0, 2.0, 3.0])
expected1 = [0.09003057, 0.24472847, 0.66524096]
assert all(abs(r - e) < 1e-6 for r, e in zip(result1, expected1))

# Test case 2: Uniform input
result2 = softmax([1.0, 1.0, 1.0])
expected2 = [0.33333333, 0.33333333, 0.33333333]
assert all(abs(r - e) < 1e-6 for r, e in zip(result2, expected2))

# Test case 3: All zeros
result3 = softmax([0.0, 0.0, 0.0])
assert all(abs(r - 0.33333333) < 1e-6 for r in result3)

# Test case 4: Probabilities sum to 1
result4 = softmax([2.0, 1.0, 0.1])
assert abs(sum(result4) - 1.0) < 1e-10

# Test case 5: Large values (numerical stability test)
result5 = softmax([1000.0, 1001.0, 1002.0])
assert all(r > 0 for r in result5)  # No overflow
assert abs(sum(result5) - 1.0) < 1e-10

# Test case 6: Negative values
result6 = softmax([-1.0, 0.0, 1.0])
assert all(r > 0 for r in result6)
assert abs(sum(result6) - 1.0) < 1e-10
```

## Time Complexity

**O(n)** where n is the length of the input vector.
- One pass to find maximum: O(n)
- One pass to compute exponentials: O(n)
- One pass to sum exponentials: O(n)
- One pass to normalize: O(n)
- Total: O(n)

## Space Complexity

**O(n)** for storing the exponentials and output vector.

## References

- [Softmax Function - Wikipedia](https://en.wikipedia.org/wiki/Softmax_function)
- [Deep Learning Book by Ian Goodfellow](https://www.deeplearningbook.org/)
- [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/linear-classify/#softmax)
