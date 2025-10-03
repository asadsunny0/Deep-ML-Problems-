# Single Gradient Descent Step

**Difficulty:** Medium  
**Category:** Calculus / Optimization  
**Tags:** [optimization, gradient-descent, derivatives]

## Problem Description

Implement a single step of gradient descent for a given function. Given a current position, a gradient (or function to compute gradient), and a learning rate, compute the new position after one gradient descent update.

## Mathematical Formulation

The gradient descent update rule is:

$$
x_{new} = x_{old} - \alpha \nabla f(x_{old})
$$

where:
- $x_{old}$ is the current position
- $\alpha$ is the learning rate (step size)
- $\nabla f(x_{old})$ is the gradient of function f at $x_{old}$
- $x_{new}$ is the updated position

For a simple quadratic function $f(x) = x^2$, the gradient is:

$$
\nabla f(x) = 2x
$$

## Constraints

- Learning rate α > 0
- x can be a scalar or a vector
- Gradient must be computed at the current position

## Examples

### Example 1
```
Input: 
x = 5.0
learning_rate = 0.1
gradient = 10.0  # (gradient at x=5 for f(x)=x^2 is 2*5=10)

Output: 
4.0

Explanation:
x_new = 5.0 - 0.1 * 10.0 = 5.0 - 1.0 = 4.0
```

### Example 2
```
Input: 
x = [1.0, 2.0, 3.0]
learning_rate = 0.01
gradient = [2.0, 4.0, 6.0]

Output: 
[0.98, 1.96, 2.94]

Explanation:
x_new[0] = 1.0 - 0.01 * 2.0 = 0.98
x_new[1] = 2.0 - 0.01 * 4.0 = 1.96
x_new[2] = 3.0 - 0.01 * 6.0 = 2.94
```

### Example 3
```
Input: 
x = 0.0
learning_rate = 0.1
gradient = 0.0

Output: 
0.0

Explanation:
At a critical point where gradient is 0, position doesn't change.
```

## Solution Approach

1. Take the current position x
2. Compute (or use provided) gradient at x
3. Multiply gradient by learning rate
4. Subtract the result from current position
5. Return the new position

This is the fundamental operation in training neural networks and many optimization algorithms.

## Implementation

### Python
```python
def gradient_descent_step(x, learning_rate, gradient):
    """
    Perform a single gradient descent step.
    
    Args:
        x: float or List[float] - Current position
        learning_rate: float - Step size (alpha)
        gradient: float or List[float] - Gradient at current position
    
    Returns:
        float or List[float] - New position after gradient descent step
    """
    # Handle scalar case
    if isinstance(x, (int, float)):
        return x - learning_rate * gradient
    
    # Handle vector case
    return [xi - learning_rate * gi for xi, gi in zip(x, gradient)]


def gradient_descent_with_function(x, learning_rate, f_gradient, num_steps=1):
    """
    Perform gradient descent steps with a gradient function.
    
    Args:
        x: float or List[float] - Starting position
        learning_rate: float - Step size
        f_gradient: callable - Function that computes gradient at x
        num_steps: int - Number of steps to perform
    
    Returns:
        float or List[float] - Final position after num_steps
    """
    current_x = x
    for _ in range(num_steps):
        grad = f_gradient(current_x)
        current_x = gradient_descent_step(current_x, learning_rate, grad)
    return current_x


# Example: Gradient for f(x) = x^2
def gradient_squared(x):
    """Gradient of f(x) = x^2 is 2x"""
    if isinstance(x, (int, float)):
        return 2 * x
    return [2 * xi for xi in x]
```

### JavaScript
```javascript
function gradientDescentStep(x, learningRate, gradient) {
    // Handle scalar case
    if (typeof x === 'number') {
        return x - learningRate * gradient;
    }
    
    // Handle vector case
    return x.map((xi, i) => xi - learningRate * gradient[i]);
}

function gradientDescentWithFunction(x, learningRate, fGradient, numSteps = 1) {
    let currentX = x;
    for (let i = 0; i < numSteps; i++) {
        const grad = fGradient(currentX);
        currentX = gradientDescentStep(currentX, learningRate, grad);
    }
    return currentX;
}

// Example: Gradient for f(x) = x^2
function gradientSquared(x) {
    if (typeof x === 'number') {
        return 2 * x;
    }
    return x.map(xi => 2 * xi);
}
```

## Test Cases

```python
# Test case 1: Single scalar step
x1 = 5.0
result1 = gradient_descent_step(x1, 0.1, 10.0)
assert abs(result1 - 4.0) < 1e-10

# Test case 2: Vector step
x2 = [1.0, 2.0, 3.0]
grad2 = [2.0, 4.0, 6.0]
result2 = gradient_descent_step(x2, 0.01, grad2)
expected2 = [0.98, 1.96, 2.94]
assert all(abs(r - e) < 1e-10 for r, e in zip(result2, expected2))

# Test case 3: Zero gradient (critical point)
x3 = 0.0
result3 = gradient_descent_step(x3, 0.1, 0.0)
assert abs(result3 - 0.0) < 1e-10

# Test case 4: Multiple steps for f(x) = x^2
x4 = 10.0
result4 = gradient_descent_with_function(x4, 0.1, gradient_squared, num_steps=5)
# Starting at 10, should approach 0
assert abs(result4) < abs(x4)

# Test case 5: Large learning rate
x5 = 1.0
result5 = gradient_descent_step(x5, 0.5, 2.0)
assert abs(result5 - 0.0) < 1e-10

# Test case 6: Negative gradient (ascending direction)
x6 = 1.0
result6 = gradient_descent_step(x6, 0.1, -5.0)
assert result6 > x6  # Should move in positive direction
```

## Time Complexity

**O(d)** where d is the dimensionality of x (number of parameters).
- For scalar: O(1)
- For vector of length d: O(d)

## Space Complexity

**O(d)** for storing the new position vector.
- For scalar: O(1)
- For vector of length d: O(d)

## References

- [Gradient Descent - Wikipedia](https://en.wikipedia.org/wiki/Gradient_descent)
- [An overview of gradient descent optimization algorithms](https://arxiv.org/abs/1609.04747)
- Sebastian Ruder, "An overview of gradient descent optimization algorithms"
- Deep Learning Book, Chapter 4: Numerical Computation
