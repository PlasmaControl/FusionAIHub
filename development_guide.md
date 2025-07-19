# Development Guide

This guide covers the development setup and coding standards for this Python package.

## Development Setup

```bash
# Create virtual environment
python -m venv .venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e "."

# Install pre-commit hooks
pre-commit install

# Run tests with coverage
pytest --cov=src --cov-report=html

# Run ruff linting
ruff check .

# Run ruff formatting
ruff format .

# Fix auto-fixable issues
ruff check --fix .
```

## Code Standards

- **Line length**: 79 characters maximum
- **Docstring format**: NumPy style
- **Linting**: Ruff
- **Function arguments**: Each argument on a new line for multi-argument functions

## Code Examples

### Function Definition Style

```python
def my_function(
    param1: str,
    param2: int,
    param3: float = 0.0,
    param4: bool = True,
) -> dict:
    """
    Brief description of the function.

    Longer description if needed. This can span multiple lines and
    should explain what the function does in more detail.

    Parameters
    ----------
    param1 : str
        Description of param1.
    param2 : int
        Description of param2.
    param3 : float, optional
        Description of param3, by default 0.0.
    param4 : bool, optional
        Description of param4, by default True.

    Returns
    -------
    dict
        Description of the return value.

    Examples
    --------
    >>> result = my_function("hello", 42)
    >>> print(result)
    {'message': 'hello', 'number': 42}
    """
    return {"message": param1, "number": param2, "value": param3}
```

### Class Definition Style

```python
class MyClass:
    """
    Brief description of the class.

    Longer description explaining the purpose and usage of the class.

    Parameters
    ----------
    name : str
        The name of the instance.
    value : int
        The initial value.

    Attributes
    ----------
    name : str
        The name of the instance.
    value : int
        The current value.

    Examples
    --------
    >>> obj = MyClass("test", 100)
    >>> obj.increment()
    >>> print(obj.value)
    101
    """

    def __init__(
        self,
        name: str,
        value: int,
    ) -> None:
        self.name = name
        self.value = value

    def increment(self) -> None:
        """Increment the value by 1."""
        self.value += 1
```

---

## Development Workflow

### 1. Setup Development Environment

```bash
# Create virtual environment
python -m venv .venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e "."
```

### 2. Running Tests

```bash
# Run tests with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_example.py

# Run with verbose output
pytest -v
```

### 3. Code Quality Checks

```bash
# Run ruff linting
ruff check .

# Run ruff formatting
ruff format .

# Fix auto-fixable issues
ruff check --fix .
```

---

## Additional Resources

- [NumPy Docstring Guide](https://numpydoc.readthedocs.io/en/latest/format.html)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)
- [pytest Documentation](https://docs.pytest.org/)