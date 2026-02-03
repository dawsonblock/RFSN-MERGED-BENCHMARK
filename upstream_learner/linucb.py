"""LinUCB contextual bandit implementation.

Pure Python implementation with Gauss-Jordan matrix inversion.
No external dependencies (no numpy/scipy required).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


def dot(a: list[float], b: list[float]) -> float:
    """Dot product of two vectors."""
    return sum(x * y for x, y in zip(a, b, strict=True))


def mat_vec(A: list[list[float]], x: list[float]) -> list[float]:
    """Matrix-vector multiplication."""
    return [dot(row, x) for row in A]


def outer(x: list[float]) -> list[list[float]]:
    """Outer product of vector with itself."""
    return [[xi * xj for xj in x] for xi in x]


def mat_add(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """Matrix addition."""
    return [[a + b for a, b in zip(r1, r2, strict=True)] for r1, r2 in zip(A, B, strict=True)]


def vec_add(a: list[float], b: list[float]) -> list[float]:
    """Vector addition."""
    return [x + y for x, y in zip(a, b, strict=True)]


def scalar_vec(s: float, x: list[float]) -> list[float]:
    """Scalar-vector multiplication."""
    return [s * xi for xi in x]


def scalar_mat(s: float, A: list[list[float]]) -> list[list[float]]:
    """Scalar-matrix multiplication."""
    return [[s * a for a in row] for row in A]


def identity(d: int) -> list[list[float]]:
    """Create dxd identity matrix."""
    return [[1.0 if i == j else 0.0 for j in range(d)] for i in range(d)]


def gauss_jordan_inverse(A: list[list[float]]) -> list[list[float]]:
    """Compute matrix inverse using Gauss-Jordan elimination.

    Falls back to identity matrix if singular.
    """
    n = len(A)
    # Augment with identity
    M = [row[:] + eye for row, eye in zip(A, identity(n), strict=True)]

    # Forward elimination
    for i in range(n):
        pivot = M[i][i]
        if abs(pivot) < 1e-12:
            # find swap
            for k in range(i + 1, n):
                if abs(M[k][i]) > 1e-12:
                    M[i], M[k] = M[k], M[i]
                    pivot = M[i][i]
                    break
        if abs(pivot) < 1e-12:
            # singular -> return identity fallback
            return identity(n)

        inv_p = 1.0 / pivot
        M[i] = [v * inv_p for v in M[i]]
        for k in range(n):
            if k == i:
                continue
            factor = M[k][i]
            if abs(factor) < 1e-12:
                continue
            M[k] = [vk - factor * vi for vk, vi in zip(M[k], M[i], strict=True)]

    return [row[n:] for row in M]


@dataclass
class LinUCBArm:
    """Single arm for LinUCB contextual bandit.

    Maintains A matrix and b vector for ridge regression.
    """

    d: int
    alpha: float = 1.25
    ridge: float = 1.0
    A: list[list[float]] = field(default_factory=list)
    b: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.A:
            self.A = mat_add(identity(self.d), scalar_mat(self.ridge, identity(self.d)))
        if not self.b:
            self.b = [0.0] * self.d

    def to_dict(self) -> dict[str, Any]:
        """Serialize arm state to dict."""
        return {
            "d": self.d,
            "alpha": self.alpha,
            "ridge": self.ridge,
            "A": self.A,
            "b": self.b,
        }

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> LinUCBArm:
        """Deserialize arm state from dict."""
        arm = cls(
            d=int(obj["d"]),
            alpha=float(obj.get("alpha", 1.25)),
            ridge=float(obj.get("ridge", 1.0)),
        )
        arm.A = obj.get("A", arm.A)
        arm.b = obj.get("b", arm.b)
        return arm

    def score(self, x: list[float]) -> float:
        """Compute UCB score for context x."""
        Ainv = gauss_jordan_inverse(self.A)
        theta = mat_vec(Ainv, self.b)
        mean = dot(theta, x)

        # sqrt(x^T A^-1 x)
        Ax = mat_vec(Ainv, x)
        var = dot(x, Ax)
        ucb = mean + self.alpha * math.sqrt(max(0.0, var))
        return ucb

    def update(self, x: list[float], r: float) -> None:
        """Update arm with reward observation."""
        self.A = mat_add(self.A, outer(x))
        self.b = vec_add(self.b, scalar_vec(r, x))
