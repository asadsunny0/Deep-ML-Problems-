# Contributing to Deep ML Problems

Thank you for your interest in contributing ML math problems to this repository! This guide will help you understand how to upload and format your problems.

## How to Contribute

### 1. Choose the Right Category

Place your problem in the appropriate category folder:
- `problems/linear_algebra/` - Matrix operations, vector spaces, eigenvalues, etc.
- `problems/calculus/` - Derivatives, integrals, gradients, chain rule, etc.
- `problems/probability/` - Probability distributions, Bayes theorem, etc.
- `problems/statistics/` - Statistical inference, hypothesis testing, etc.
- `problems/optimization/` - Gradient descent, convex optimization, etc.
- `problems/neural_networks/` - Backpropagation, activation functions, etc.

### 2. Use the Problem Template

Use the `PROBLEM_TEMPLATE.md` file as a starting point for your problem. This ensures consistency across all problems.

### 3. Problem Naming Convention

Name your problem file descriptively using snake_case:
```
matrix_multiplication.md
gradient_descent_step.md
softmax_implementation.md
```

### 4. Problem Structure

A good problem should include:
- **Clear problem statement**: What needs to be solved?
- **Mathematical formulation**: Relevant equations and formulas
- **Examples**: At least 2 examples with explanations
- **Solution approach**: Brief explanation of how to solve it
- **Implementation**: Working code in at least one language (Python preferred)
- **Test cases**: Verify the solution works correctly
- **Complexity analysis**: Time and space complexity

### 5. Quality Guidelines

- **Clarity**: Write clear, concise problem descriptions
- **Correctness**: Ensure all mathematical formulas are correct
- **Completeness**: Include all sections from the template
- **Testing**: Verify your solution works with multiple test cases
- **Formatting**: Use proper markdown formatting and LaTeX for equations

### 6. Code Style

**Python:**
- Follow PEP 8 style guidelines
- Use descriptive variable names
- Add comments for complex logic
- Include type hints where appropriate

**JavaScript:**
- Follow standard JavaScript conventions
- Use meaningful variable names
- Add JSDoc comments

### 7. Mathematical Notation

Use LaTeX for mathematical expressions:
- Inline math: `$expression$`
- Display math: `$$expression$$`

Examples:
```markdown
The gradient is calculated as: $\nabla f(x) = \frac{\partial f}{\partial x}$

$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}
$$
```

### 8. Submitting Your Problem

1. Fork the repository
2. Create a new branch: `git checkout -b add-problem-name`
3. Add your problem file to the appropriate category folder
4. Commit your changes: `git commit -m "Add [problem name] problem"`
5. Push to your fork: `git push origin add-problem-name`
6. Create a Pull Request

### 9. Review Process

Your submission will be reviewed for:
- Correctness of mathematics and code
- Completeness of the problem description
- Code quality and testing
- Appropriate difficulty level and categorization

## Need Help?

If you have questions or need help:
- Open an issue with your question
- Check existing problems for examples
- Refer to the PROBLEM_TEMPLATE.md

## Code of Conduct

- Be respectful and constructive in all interactions
- Give credit where credit is due
- Help others learn and improve

Thank you for contributing to the Deep ML Problems repository!
