"""CI Entrypoint - GitHub Action Integration (Unified Architecture).

Adapts the Controller to run within a CI environment (GitHub Actions) using the
Unified Evaluation Harness (eval.run_v2).

Reads inputs from environment variables, executes the benchmark, and 
outputs results to GITHUB_OUTPUT and GITHUB_STEP_SUMMARY.

Enforces RFSN_BENCH_STRICT=1.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rfsn_controller.structured_logging import get_logger

# Import Unified Harness
from eval.run_v2 import run_eval, EvalResult
from eval.strictness import strict_benchmark_mode

logger = get_logger(__name__)


@dataclass
class CIBenchmarkConfig:
    """Configuration for CI benchmark runs."""
    
    dataset: str = "swebench_lite"
    max_tasks: int | None = None
    output_dir: Path = Path("runs")
    results_dir: Path = Path("eval_results")
    strict_mode: bool = True
    model: str = "deepseek"
    
    @classmethod
    def from_env(cls) -> "CIBenchmarkConfig":
        """Load configuration from environment variables."""
        max_tasks_str = os.environ.get("INPUT_MAX_TASKS", "0")
        max_tasks = int(max_tasks_str) if max_tasks_str != "0" else None
        
        return cls(
            dataset=os.environ.get("INPUT_DATASET", "swebench_lite.jsonl"),
            max_tasks=max_tasks,
            output_dir=Path(os.environ.get("INPUT_OUTPUT_DIR", "runs")),
            results_dir=Path(os.environ.get("INPUT_RESULTS_DIR", "eval_results")),
            strict_mode=os.environ.get("RFSN_STRICT_BENCH", "").lower() in {"1", "true", "yes"},
            model=os.environ.get("INPUT_MODEL", "deepseek"),
        )


def write_github_output(key: str, value: str) -> None:
    """Write a value to GITHUB_OUTPUT."""
    output_file = os.environ.get("GITHUB_OUTPUT")
    if output_file:
        with open(output_file, "a") as f:
            f.write(f"{key}={value}\n")
    else:
        logger.info(f"[CI Output] {key}={value}")


def write_step_summary(markdown: str) -> None:
    """Write markdown summary to GITHUB_STEP_SUMMARY."""
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a") as f:
            f.write(markdown)
    else:
        print("[CI Summary]")
        print(markdown)


def generate_results_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Generate a summary of benchmark results."""
    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total": len(results),
        "passed": sum(1 for r in results if r.get("passed")),
        "failed": sum(1 for r in results if not r.get("passed")),
        "by_status": {},
        "avg_runtime": 0.0,
        "total_attempts": 0,
        "total_gate_rejections": 0,
        "total_security_violations": 0,
    }
    
    for r in results:
        # Count by status
        status = r.get("status", "UNKNOWN")
        summary["by_status"][status] = summary["by_status"].get(status, 0) + 1
        
        # Aggregate metrics
        summary["total_attempts"] += r.get("attempts", 0)
        summary["total_gate_rejections"] += r.get("gate_rejections", 0)
        summary["total_security_violations"] += r.get("security_violations", 0)
    
    # Calculate averages
    if results:
        total_time = sum(r.get("runtime", 0) for r in results)
        summary["avg_runtime"] = total_time / len(results)
    
    return summary


def run_ci_benchmark(config: CIBenchmarkConfig) -> int:
    """Run RFSN benchmark in CI mode."""
    start_time = time.time()
    
    # Enforce strict mode
    if config.strict_mode and not strict_benchmark_mode():
        logger.error("FATAL: RFSN_BENCH_STRICT must be set for benchmark mode")
        write_github_output("success", "false")
        write_github_output("error", "strict_mode_not_set")
        return 1
    
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"[CI] Starting benchmark: dataset={config.dataset}")
    logger.info(f"[CI] Max tasks: {config.max_tasks or 'all'}")
    logger.info(f"[CI] Model: {config.model}")
    
    try:
        from agent.llm_patcher import get_llm_patch_fn
        
        # Run unified evaluation
        results_objs = run_eval(
            dataset_name=config.dataset,
            max_tasks=config.max_tasks,
            llm_patch_fn=get_llm_patch_fn(config.model),
            max_attempts=6,  # Standard budget
            results_dir=str(config.results_dir),
        )
        
        # Convert to dicts
        result_dicts = [r.to_dict() for r in results_objs]
        
        # Generate summary
        summary = generate_results_summary(result_dicts)
        summary["dataset"] = config.dataset
        summary["strict_mode"] = True
        summary["total_runtime"] = time.time() - start_time
        
        # Write results.json
        results_json_path = Path("results.json")
        with open(results_json_path, "w") as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"[CI] Wrote summary to {results_json_path}")
        
        # Write GitHub outputs
        pass_rate = (summary["passed"] / max(1, summary["total"]) * 100)
        write_github_output("success", str(summary["passed"] == summary["total"]).lower())
        write_github_output("total_tasks", str(summary["total"]))
        write_github_output("passed_tasks", str(summary["passed"]))
        write_github_output("failed_tasks", str(summary["failed"]))
        write_github_output("pass_rate", f"{pass_rate:.1f}")
        
        # Write step summary
        summary_md = f"""## RFSN Benchmark Results (Unified v2)

| Metric | Value |
|--------|-------|
| Dataset | {config.dataset} |
| Total Tasks | {summary['total']} |
| Passed | {summary['passed']} |
| Failed | {summary['failed']} |
| Pass Rate | {pass_rate:.1f}% |
| Total Runtime | {summary['total_runtime']:.1f}s |
| Avg Attempts | {summary['total_attempts'] / max(1, summary['total']):.1f} |
| Gate Rejections | {summary['total_gate_rejections']} |
| Security Violations | {summary['total_security_violations']} |

### Status Breakdown

"""
        for status, count in sorted(summary["by_status"].items()):
            summary_md += f"- **{status}**: {count}\n"
            
        if summary["total_security_violations"] > 0:
            summary_md += f"\n### ⚠️ Security Violations: {summary['total_security_violations']}\n"
            
        write_step_summary(summary_md)
        
        # Success if 100% pass in strict mode?
        # Typically benchmark is reporting.
        # But if strict mode requires all pass?
        # The legacy code returned 0 even on failure, unless SystemError.
        
        return 0
        
    except Exception as e:
        logger.exception(f"[CI] Benchmark failed with error: {e}")
        write_github_output("success", "false")
        write_github_output("error", str(e)[:200])
        return 1


def main() -> int:
    """Main entry point."""
    config = CIBenchmarkConfig.from_env()
    return run_ci_benchmark(config)


if __name__ == "__main__":
    sys.exit(main())
