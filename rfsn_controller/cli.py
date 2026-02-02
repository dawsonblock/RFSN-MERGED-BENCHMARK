from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv

from .controller import ControllerConfig, run_controller
from .performance import start_prewarm_background


# CGW mode import (lazy to avoid circular imports)
def _run_cgw_mode(args):
    """Run in CGW serial decision mode."""
    from .cgw_cli import run_cgw_agent
    return run_cgw_agent(
        github_url=args.repo,
        test_cmd=args.test,
        max_cycles=getattr(args, "max_cgw_cycles", 50),
        max_patches=getattr(args, "max_patches", 10),
        verbose=getattr(args, "verbose", False),
    )


def main() -> None:
    """Entry point for the CLI.

    Loads environment variables from .env, parses command-line arguments,
    constructs a ControllerConfig, and runs the controller.
    """
    load_dotenv()
    
    # Start Docker image pre-warming in background for faster cold starts
    start_prewarm_background()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        required=True,
        help="Public GitHub URL: https://github.com/OWNER/REPO",
    )
    parser.add_argument(
        "--test",
        default="pytest -q",
        help="Test command to satisfy",
    )
    parser.add_argument(
        "--ref",
        default=None,
        help="Optional branch/tag/sha to checkout before running",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=12,
        help="Maximum controller steps before giving up",
    )
    parser.add_argument(
        "--fix-all",
        action="store_true",
        help="Continue fixing bugs until all tests pass (no max steps limit)",
    )
    parser.add_argument(
        "--max-steps-without-progress",
        type=int,
        default=10,
        help="Early termination if no progress after N steps (default: 10)",
    )
    parser.add_argument(
        "--collect-finetuning-data",
        action="store_true",
        help="Collect successful patches for model fine-tuning",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("RFSN_MODEL", "deepseek-chat"),
        help="Model to use (default: deepseek-chat, or RFSN_MODEL env var)",
    )
    parser.add_argument(
        "--time-mode",
        default="frozen",
        choices=["frozen", "live"],
        help="Time mode for determinism (default: frozen)",
    )
    parser.add_argument(
        "--run-started-at-utc",
        default=None,
        help="Optional ISO-8601 UTC timestamp for deterministic replay",
    )
    parser.add_argument(
        "--time-seed",
        type=int,
        default=None,
        help="Optional integer seed used for deterministic run metadata",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=None,
        help=("Optional integer seed for Python/NumPy RNG (deterministic replay)"),
    )
    parser.add_argument(
        "--max-minutes",
        type=int,
        default=30,
        help="Total time budget in minutes (default: 30)",
    )
    parser.add_argument(
        "--install-timeout",
        type=int,
        default=300,
        help="Timeout for dependency installation in seconds (default: 300)",
    )
    parser.add_argument(
        "--focus-timeout",
        type=int,
        default=120,
        help="Timeout for focused test runs in seconds (default: 120)",
    )
    parser.add_argument(
        "--full-timeout",
        type=int,
        default=300,
        help="Timeout for full test suite runs in seconds (default: 300)",
    )
    parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=40,
        help="Maximum total tool calls per run (default: 40)",
    )
    parser.add_argument(
        "--docker-image",
        default="python:3.11-slim",
        help=("Docker image for sandboxed execution (default: python:3.11-slim)"),
    )
    parser.add_argument(
        "--unsafe-host-exec",
        action="store_true",
        help=("Allow running commands on host instead of Docker (DANGEROUS, not recommended)"),
    )
    parser.add_argument(
        "--cpu",
        type=float,
        default=2.0,
        help="Docker CPU limit (default: 2.0)",
    )
    parser.add_argument(
        "--mem-mb",
        type=int,
        default=4096,
        help="Docker memory limit in MB (default: 4096)",
    )
    parser.add_argument(
        "--pids",
        type=int,
        default=256,
        help="Docker process ID limit (default: 256)",
    )
    parser.add_argument(
        "--docker-readonly",
        action="store_true",
        help="Mount repo as read-only with /tmp as tmpfs (more secure)",
    )
    parser.add_argument(
        "--lint-cmd",
        default=None,
        help="Lint command for verification (e.g., 'ruff check .')",
    )
    parser.add_argument(
        "--typecheck-cmd",
        default=None,
        help="Typecheck command for verification (e.g., 'mypy .')",
    )
    parser.add_argument(
        "--repro-cmd",
        default=None,
        help="Repro command for verification (e.g., 'pytest -q --repeat=2')",
    )
    parser.add_argument(
        "--verify-cmd",
        default=None,
        help="Smoke test command for feature verification (e.g., './run_smoke_tests.sh')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=("Clone + detect + setup + baseline test, then exit (no repair loop)"),
    )
    parser.add_argument(
        "--project-type",
        default="auto",
        choices=["auto", "python", "node", "go", "rust", "java", "dotnet"],
        help="Project type (default: auto-detect)",
    )
    parser.add_argument(
        "--buildpack",
        default="auto",
        help=("Docker buildpack image (default: auto-select based on project type)"),
    )
    parser.add_argument(
        "--enable-sysdeps",
        action="store_true",
        help="Enable automatic system dependency installation (SYSDEPS phase)",
    )
    parser.add_argument(
        "--sysdeps-tier",
        default="4",
        choices=["0", "1", "2", "3", "4", "5", "6", "7"],
        help="Maximum APT tier for system dependencies (default: 4)",
    )
    parser.add_argument(
        "--sysdeps-max-packages",
        type=int,
        default=10,
        help="Maximum number of system packages to install (default: 10)",
    )
    parser.add_argument(
        "--build-cmd",
        default=None,
        help="Build command for verification (e.g., 'npm run build')",
    )
    parser.add_argument(
        "--learning-db",
        default=None,
        help=("Path to SQLite DB for controller-owned learning (disabled if omitted)"),
    )
    parser.add_argument(
        "--learning-half-life-days",
        type=int,
        default=14,
        help=("Decay half-life for learning weights in record-age units (default: 14)"),
    )
    parser.add_argument(
        "--learning-max-age-days",
        type=int,
        default=90,
        help=("Drop learning records older than N record-age units (default: 90)"),
    )
    parser.add_argument(
        "--learning-max-rows",
        type=int,
        default=20000,
        help="Maximum number of learning rows to retain (default: 20000)",
    )
    parser.add_argument(
        "--feature-mode",
        action="store_true",
        help="Enable feature engineering mode (vs repair mode)",
    )
    parser.add_argument(
        "--feature-description",
        default=None,
        help="Feature specification for feature mode",
    )
    parser.add_argument(
        "--acceptance-criteria",
        action="append",
        help="Acceptance criteria for feature mode (can be specified multiple times)",
    )
    parser.add_argument(
        "--verify-policy",
        default="tests_only",
        choices=["tests_only", "cmds_then_tests", "cmds_only"],
        help="Verification policy for feature mode (default: tests_only)",
    )
    parser.add_argument(
        "--focused-verify-cmd",
        action="append",
        dest="focused_verify_cmds",
        help="Focused verification command (can be specified multiple times)",
    )
    parser.add_argument(
        "--verify-cmd-extra",
        action="append",
        dest="verify_cmds",
        help="Additional verification command (can be specified multiple times)",
    )
    parser.add_argument(
        "--max-lines-changed",
        type=int,
        default=None,
        help="Override maximum lines changed in patches",
    )
    parser.add_argument(
        "--max-files-changed",
        type=int,
        default=None,
        help="Override maximum files changed in patches",
    )
    parser.add_argument(
        "--allow-lockfile-changes",
        action="store_true",
        help="Allow patches to modify lockfiles (package-lock.json, yarn.lock, etc.)",
    )
    parser.add_argument(
        "--enable-llm-cache",
        action="store_true",
        help="Enable LLM response caching for faster re-runs (default: disabled)",
    )
    parser.add_argument(
        "--llm-cache-path",
        default=None,
        help="Path to LLM cache database (default: ~/.rfsn/llm_cache.db)",
    )
    parser.add_argument(
        "--parallel-patches",
        action="store_true",
        help="Generate patches at multiple temperatures in parallel (faster)",
    )
    parser.add_argument(
        "--no-prewarm",
        action="store_true",
        help="Disable Docker image pre-warming",
    )
    parser.add_argument(
        "--ensemble-mode",
        action="store_true",
        help="Use multi-model ensemble for improved patch quality",
    )
    parser.add_argument(
        "--incremental-tests",
        action="store_true",
        help="Run only affected tests first for faster feedback",
    )
    parser.add_argument(
        "--enable-telemetry",
        action="store_true",
        help="Enable OpenTelemetry tracing and Prometheus metrics",
    )
    parser.add_argument(
        "--telemetry-port",
        type=int,
        default=8080,
        help="Port for Prometheus metrics endpoint (default: 8080)",
    )
    # Elite Controller flags
    parser.add_argument(
        "--policy-mode",
        default="off",
        choices=["off", "bandit"],
        help="Learning policy mode: off or bandit (Thompson Sampling)",
    )
    parser.add_argument(
        "--planner-mode",
        default="off",
        choices=["off", "v2", "dag", "v5"],
        help="Planner mode: off|v2|dag|v5 (v5 is latest with meta-planning and state tracking)",
    )
    parser.add_argument(
        "--risk-profile",
        default="production",
        choices=["production", "research"],
        help="Risk profile: production (strict) or research (tournament-friendly). State/memory are separated.",
    )
    parser.add_argument(
        "--state-dir",
        default=None,
        help="Host directory for persistent state/logs (will create <state_dir>/<risk_profile>/<run_id>/).",
    )
    parser.add_argument(
        "--durability-reruns",
        type=int,
        default=0,
        help="After a patch passes full tests, rerun full tests N more times to detect flakiness (default: 0).",
    )
    parser.add_argument(
        "--repo-index",
        action="store_true",
        help="Enable repository indexing for context",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Deterministic seed for reproducibility (default: 1337)",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip final evaluation phase",
    )
    # CGW Serial Decision Mode
    parser.add_argument(
        "--cgw-mode",
        action="store_true",
        help="Use CGW serial decision architecture (one decision per cycle)",
    )
    parser.add_argument(
        "--max-cgw-cycles",
        type=int,
        default=50,
        help="Maximum decision cycles in CGW mode (default: 50)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    # Beam Search options
    parser.add_argument(
        "--beam-search",
        action="store_true",
        help="Enable multi-step beam search over patch space",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=3,
        help="Number of candidates to keep per beam search step (default: 3)",
    )
    parser.add_argument(
        "--beam-depth",
        type=int,
        default=5,
        help="Maximum beam search expansion depth (default: 5)",
    )
    parser.add_argument(
        "--beam-score-threshold",
        type=float,
        default=0.95,
        help="Score threshold for early termination in beam search (default: 0.95)",
    )
    parser.add_argument(
        "--beam-timeout",
        type=float,
        default=300.0,
        help="Timeout in seconds for beam search (default: 300)",
    )
    args = parser.parse_args()

    # Handle CGW mode
    if args.cgw_mode:
        result = _run_cgw_mode(args)
        import sys
        sys.exit(0 if result.get("success") else 1)

    cfg = ControllerConfig(
        github_url=args.repo,
        test_cmd=args.test,
        ref=args.ref,
        max_steps=args.steps,
        fix_all=args.fix_all,
        max_steps_without_progress=args.max_steps_without_progress,
        collect_finetuning_data=args.collect_finetuning_data,
        model=args.model,
        time_mode=args.time_mode,
        run_started_at_utc=args.run_started_at_utc,
        time_seed=args.time_seed,
        rng_seed=args.rng_seed,
        max_minutes=args.max_minutes,
        install_timeout=args.install_timeout,
        focus_timeout=args.focus_timeout,
        full_timeout=args.full_timeout,
        max_tool_calls=args.max_tool_calls,
        docker_image=args.docker_image,
        unsafe_host_exec=args.unsafe_host_exec,
        cpu=args.cpu,
        mem_mb=args.mem_mb,
        pids=args.pids,
        docker_readonly=args.docker_readonly,
        lint_cmd=args.lint_cmd,
        typecheck_cmd=args.typecheck_cmd,
        repro_cmd=args.repro_cmd,
        verify_cmd=args.verify_cmd,
        dry_run=args.dry_run,
        project_type=args.project_type,
        buildpack=args.buildpack,
        enable_sysdeps=args.enable_sysdeps,
        sysdeps_tier=int(args.sysdeps_tier),
        sysdeps_max_packages=args.sysdeps_max_packages,
        build_cmd=args.build_cmd,
        learning_db_path=args.learning_db,
        learning_half_life_days=args.learning_half_life_days,
        learning_max_age_days=args.learning_max_age_days,
        learning_max_rows=args.learning_max_rows,
        feature_mode=args.feature_mode,
        feature_description=args.feature_description,
        acceptance_criteria=args.acceptance_criteria or [],
        verify_policy=args.verify_policy,
        focused_verify_cmds=args.focused_verify_cmds or [],
        verify_cmds=args.verify_cmds or [],
        max_lines_changed=args.max_lines_changed,
        max_files_changed=args.max_files_changed,
        allow_lockfile_changes=args.allow_lockfile_changes,
        enable_llm_cache=args.enable_llm_cache,
        llm_cache_path=args.llm_cache_path,
        parallel_patches=args.parallel_patches,
        ensemble_mode=args.ensemble_mode,
        incremental_tests=args.incremental_tests,
        enable_telemetry=args.enable_telemetry,
        telemetry_port=args.telemetry_port,
        policy_mode=args.policy_mode,
        planner_mode=args.planner_mode,
        repo_index=args.repo_index,
        seed=args.seed,
        risk_profile=args.risk_profile,
        state_dir=args.state_dir,
        durability_reruns=args.durability_reruns,
        no_eval=args.no_eval,
        beam_search_enabled=args.beam_search,
        beam_width=args.beam_width,
        beam_depth=args.beam_depth,
        beam_score_threshold=args.beam_score_threshold,
        beam_timeout_seconds=args.beam_timeout,
    )
    result = run_controller(cfg)
    print(result)


if __name__ == "__main__":
    main()
