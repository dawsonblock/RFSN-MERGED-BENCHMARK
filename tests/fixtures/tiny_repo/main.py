# Tiny repo fixture for testing
"""Main module for the tiny test repo."""

def hello_world() -> str:
    """Return a greeting."""
    return "Hello, World!"


def add_numbers(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


class Calculator:
    """A simple calculator class."""
    
    def __init__(self, value: int = 0) -> None:
        self.value = value
    
    def add(self, x: int) -> "Calculator":
        """Add to the current value."""
        self.value += x
        return self
    
    def subtract(self, x: int) -> "Calculator":
        """Subtract from the current value."""
        self.value -= x
        return self
