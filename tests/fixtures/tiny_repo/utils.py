"""Utilities module that imports from main."""

from main import Calculator, add_numbers


def compute_sum(numbers: list) -> int:
    """Compute sum using add_numbers."""
    total = 0
    for n in numbers:
        total = add_numbers(total, n)
    return total


def make_calculator(start: int = 0) -> Calculator:
    """Create a new calculator instance."""
    return Calculator(start)
