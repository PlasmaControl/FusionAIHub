# Development Guide

This guide covers the development setup and coding standards for this Python package.

## Code Standards

- **Line length**: 79 characters maximum
- **Docstring format**: NumPy style
- **Linting**: Ruff
- **Function arguments**: Each argument on a new line for multi-argument functions

## IDE Setup

### PyCharm Setup

#### 1. Install Ruff Plugin

1. Go to **File → Settings**
2. Navigate to **Plugins**
3. Search for "Ruff" and install the official Ruff plugin
4. Restart PyCharm

#### 2. Configure Code Style

1. Go to **File → Settings → Editor → Code Style → Python**
2. Set **Hard wrap at: ** to `79`
3. In the **Wrapping and Braces** tab:
   - Set **Method declaration parameters** to "Chop down if long"
   - Check "New line after '('" and "')' on new line"
   - Set **Function call arguments** to "Chop down if long"

#### 3. Configure Docstring Format

1. Go to **File → Settings → Tools → Python Integrated Tools**
2. Set **Docstring format** to "NumPy"

#### 4. Configure Ruff

1. Go to **File → Settings → Tools → Ruff**
2. Enable **Use Ruff**
3. Set the **Ruff executable** path (if not auto-detected)
4. Enable **Run Ruff when files are saved**

### VSCode Setup

#### 1. Install Extensions

Install these extensions from the VSCode marketplace:

- **Ruff** (charliermarsh.ruff)
- **Python** (ms-python.python)
- **Python Docstring Generator** (njpwerner.autodocstring)

#### 2. Configure Settings

Create or update `.vscode/settings.json` in your project root:

```json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "editor.rulers": [79],
  "editor.wordWrap": "wordWrapColumn",
  "editor.wordWrapColumn": 79,
  
  // Ruff configuration
  "ruff.enable": true,
  "ruff.organizeImports": true,
  "ruff.fixAll": true,
  "ruff.codeAction.fixViolation": {
    "enable": true
  },
  
  // Python formatting
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.organizeImports": "explicit",
      "source.fixAll": "explicit"
    }
  },
  
  // Docstring configuration
  "autoDocstring.docstringFormat": "numpy",
  "autoDocstring.startOnNewLine": true,
  "autoDocstring.includeExtendedSummary": true,
  "autoDocstring.includeName": false,
  
  // Python specific settings
  "python.formatting.provider": "none",
  "python.linting.enabled": false,
  "python.analysis.typeCheckingMode": "basic"
}
```

#### 3. Configure Ruff

Create `ruff.toml` in your project root:

```toml
line-length = 79
target-version = "py38"

[lint]
select = [
    "E",  # pycodestyle errors
    "W",  # pycodestyle warnings
    "F",  # pyflakes
    "I",  # isort
    "B",  # flake8-bugbear
    "C4", # flake8-comprehensions
    "UP", # pyupgrade
]
ignore = [
    "E501", # line too long (handled by formatter)
]

[format]
quote-style = "double"
indent-style = "space"
skip-magic-trailing-comma = false
line-ending = "auto"
```

---

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