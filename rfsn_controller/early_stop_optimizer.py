"""Early stopping optimizer for test execution.

Stops test execution early if critical failures are detected,
saving time on patches that clearly won't work.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class EarlyStopConfig:
    """Configuration for early stopping."""
    
    enabled: bool = True
    max_failures: int = 3  # Stop after this many failures
    stop_on_syntax_error: bool = True
    stop_on_import_error: bool = True
    stop_on_critical_error: bool = True
    critical_error_patterns: list[str] = None
    
    def __post_init__(self):
        if self.critical_error_patterns is None:
            self.critical_error_patterns = [
                r"SyntaxError",
                r"IndentationError",
                r"ImportError",
                r"ModuleNotFoundError",
                r"NameError.*not defined",
                r"FATAL",
                r"segmentation fault",
                r"core dumped",
            ]


class EarlyStopDecision:
    """Decision on whether to stop early."""
    
    def __init__(
        self,
        should_stop: bool,
        reason: str,
        failures_detected: int = 0,
        error_type: str | None = None
    ):
        self.should_stop = should_stop
        self.reason = reason
        self.failures_detected = failures_detected
        self.error_type = error_type


class EarlyStopOptimizer:
    """Optimizer that decides when to stop test execution early."""
    
    def __init__(self, config: EarlyStopConfig | None = None):
        self.config = config or EarlyStopConfig()
        self._failure_count = 0
        self._critical_errors = []
    
    def reset(self) -> None:
        """Reset state for new test run."""
        self._failure_count = 0
        self._critical_errors = []
    
    def should_stop_early(
        self,
        output: str,
        stderr: str = "",
        exit_code: int = 0,
    ) -> EarlyStopDecision:
        """Determine if test execution should stop early.
        
        Args:
            output: Test stdout
            stderr: Test stderr  
            exit_code: Process exit code
            
        Returns:
            Decision object indicating whether to stop
        """
        if not self.config.enabled:
            return EarlyStopDecision(False, "Early stopping disabled")
        
        combined_output = output + "\n" + stderr
        
        # Check for critical errors
        if self.config.stop_on_critical_error:
            for pattern in self.config.critical_error_patterns:
                if re.search(pattern, combined_output, re.IGNORECASE):
                    error_type = pattern.replace("\\", "")
                    self._critical_errors.append(error_type)
                    return EarlyStopDecision(
                        True,
                        f"Critical error detected: {error_type}",
                        self._failure_count,
                        error_type
                    )
        
        # Check for syntax errors
        if self.config.stop_on_syntax_error:
            if "SyntaxError" in combined_output or "IndentationError" in combined_output:
                return EarlyStopDecision(
                    True,
                    "Syntax error detected - patch is malformed",
                    self._failure_count,
                    "SyntaxError"
                )
        
        # Check for import errors
        if self.config.stop_on_import_error:
            if "ImportError" in combined_output or "ModuleNotFoundError" in combined_output:
                return EarlyStopDecision(
                    True,
                    "Import error detected - patch breaks dependencies",
                    self._failure_count,
                    "ImportError"
                )
        
        # Count failures based on test framework output
        new_failures = self._count_failures(combined_output)
        self._failure_count += new_failures
        
        # Check if we've exceeded max failures
        if self._failure_count >= self.config.max_failures:
            return EarlyStopDecision(
                True,
                f"Max failures reached ({self._failure_count}/{self.config.max_failures})",
                self._failure_count,
                "MaxFailures"
            )
        
        return EarlyStopDecision(
            False,
            f"Continue (failures: {self._failure_count}/{self.config.max_failures})",
            self._failure_count
        )
    
    def _count_failures(self, output: str) -> int:
        """Count test failures in output."""
        failures = 0
        
        # pytest format: "FAILED test_file.py::test_name"
        failures += len(re.findall(r'^FAILED ', output, re.MULTILINE))
        
        # pytest summary: "X failed"
        match = re.search(r'(\d+) failed', output)
        if match:
            return int(match.group(1))
        
        # unittest format: "FAIL: test_name"
        failures += len(re.findall(r'^FAIL: ', output, re.MULTILINE))
        
        # unittest format: "ERROR: test_name"
        failures += len(re.findall(r'^ERROR: ', output, re.MULTILINE))
        
        return failures
    
    def get_stats(self) -> dict:
        """Get optimizer statistics."""
        return {
            'total_failures': self._failure_count,
            'critical_errors': self._critical_errors,
            'critical_error_count': len(self._critical_errors),
        }


# Global optimizer instance
_global_optimizer: EarlyStopOptimizer | None = None


def get_early_stop_optimizer() -> EarlyStopOptimizer:
    """Get global early stop optimizer."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = EarlyStopOptimizer()
    return _global_optimizer
