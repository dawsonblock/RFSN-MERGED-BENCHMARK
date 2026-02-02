"""The RFSN controller core loop.

This module implements the controller loop for the RFSN coding agent.
It manages a disposable sandbox, clones a public GitHub repository,
runs test commands to measure progress, consults the Gemini model for
tool requests or candidate patches, executes requested tools, validates
candidate patches in isolated worktrees, and applies winners to the
main repository only if they pass focused and full verification.
"""

from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass, field
from datetime import UTC
from pathlib import Path
from typing import Any

from .action_outcome_memory import (
    ActionOutcomeStore,
    format_action_priors,
    make_action_json_for_patch,
    make_action_key_for_patch,
    make_action_key_for_tool,
    make_context_signature,
    score_action,
)
from .apt_whitelist import AptTier, AptWhitelist
from .broadcaster import ProgressBroadcaster
from .buildpacks import (
    BuildpackContext,
    BuildpackType,
    get_all_buildpacks,
    get_buildpack,
)
from .clock import FrozenClock, SystemClock, make_run_id, parse_utc_iso
from .diff_minimizer import DiffMinimizer

# Optional: Evidence pack (can be removed for minimal install)
try:
    from .evidence_pack import EvidencePackConfig, EvidencePackExporter
    HAS_EVIDENCE_PACK = True
except ImportError:
    HAS_EVIDENCE_PACK = False
    EvidencePackConfig = None  # type: ignore
    EvidencePackExporter = None  # type: ignore
from .goals import DEFAULT_FEATURE_SUBGOALS
from .llm import call_ensemble_sync
from .log import write_jsonl
from .parallel import evaluate_patches_parallel, find_first_successful_patch
from .parsers import normalize_test_path, parse_trace_files
from .patch_budget import create_patch_budget_controller
from .patch_hygiene import PatchHygieneConfig, validate_patch_hygiene
from .phases import Phase, PhaseTransition
from .policy import choose_policy
from .project_detection import detect_project_type, get_default_test_command, get_setup_commands
from .prompt import MODE_FEATURE, build_model_input
from .qa import QAConfig, QAOrchestrator
from .retrieval_context import build_retrieval_context
from .sandbox import (
    DockerResult,
    Sandbox,
    apply_patch,
    apply_patch_in_dir,
    checkout,
    clone_public_github,
    create_sandbox,
    create_venv,
    docker_install,
    docker_run,
    docker_test,
    drop_worktree,
    find_local_module,
    git_status,
    grep,
    list_tree,
    make_worktree,
    pip_install,
    pip_install_progressive,
    pip_install_requirements,
    read_file,
    reset_hard,
    run_cmd,
    set_pythonpath,
)

# Optional: Setup report (can be removed for minimal install)
try:
    from .setup_report import create_setup_report
    HAS_SETUP_REPORT = True
except ImportError:
    HAS_SETUP_REPORT = False
    create_setup_report = None  # type: ignore

from .stall_detector import StallState
from .sysdeps_installer import SysdepsInstaller
from .tool_manager import ToolRequestConfig, ToolRequestManager
from .url_validation import validate_github_url
from .verifier import TestDeltaTracker, VerifyResult, run_tests


def get_model_client(model: str) -> Any:
    """Get a model client for the specified model.
    
    This is a factory function that returns an appropriate LLM client
    based on the model name. Used by the controller to abstract model
    access.
    
    Args:
        model: The model name (e.g., "deepseek-chat", "gemini-3.0-flash").
        
    Returns:
        A callable model client that can be used to make LLM calls.
        
    Raises:
        RuntimeError: If the required SDK is not available.
    """
    if "gemini" in model.lower():
        from . import llm_gemini
        return llm_gemini.call_model
    elif "deepseek" in model.lower():
        from . import llm_deepseek
        return llm_deepseek.call_model
    else:
        # Default to deepseek for unknown models
        from . import llm_deepseek
        return llm_deepseek.call_model


def _truncate(s: str, limit: int) -> str:
    """Truncate string to limit."""
    if not s:
        return ""
    if len(s) <= limit:
        return s
    return s[:limit] + "...[truncated]"





def _infer_buildpack_type_from_test_cmd(test_cmd: str) -> BuildpackType | None:
    cmd = (test_cmd or "").strip().lower()
    if not cmd:
        return None
    if (
        cmd.startswith("pytest")
        or " pytest" in cmd
        or "python -m pytest" in cmd
        or "python3 -m pytest" in cmd
    ):
        return BuildpackType.PYTHON
    if cmd.startswith(("npm ", "yarn ", "pnpm ", "npx ", "bun ")):
        return BuildpackType.NODE
    if cmd.startswith(("go test", "go test ")):
        return BuildpackType.GO
    if cmd.startswith(("cargo test", "cargo test ")):
        return BuildpackType.RUST
    if cmd.startswith(("mvn ", "./gradlew", "gradle ", "./mvnw")):
        return BuildpackType.JAVA
    if cmd.startswith(("dotnet test", "dotnet test ")):
        return BuildpackType.DOTNET
    return None


FORBIDDEN_PREFIXES = [".git/", "node_modules/", ".venv/", "venv/", "__pycache__/"]


def _diff_hash(d: str) -> str:
    """Compute a hash of a diff string for deduplication."""
    return hashlib.sha256((d or "").encode("utf-8", errors="ignore")).hexdigest()


def _safe_path(p: str) -> bool:
    """Return True if the relative path is outside forbidden prefixes."""
    p = p.replace("\\", "/").lstrip("./")
    return not any(p.startswith(pref) for pref in FORBIDDEN_PREFIXES)


def _files_block(files: list[dict[str, Any]]) -> str:
    """Create a files block for the model input from a list of read_file results."""
    blocks = []
    for f in files:
        content = f.get("content") if isinstance(f.get("content"), str) else f.get("text")
        if f.get("ok") and f.get("path") and isinstance(content, str):
            blocks.append(f"[path: {f['path']}]\n{content}\n")
    return "\n".join(blocks)


def _constraints_text() -> str:
    """Return a static constraints description for the model."""
    return "\n".join(
        [
            "- Return either tool_request or patch JSON only.",
            "- Patch diff must apply with git apply from repo root.",
            "- Minimal edits. No refactors. No reformatting.",
            "- Public GitHub only. No tokens.",
            "- Do not touch forbidden paths: " + ", ".join(FORBIDDEN_PREFIXES),
            "- IMPORTANT: Commands run with shell=False - no shell features allowed:",
            "  * No command chaining (&&, ||, ;)",
            "  * No pipes (|) or redirects (>, <, >>)",
            "  * No command substitution ($(), backticks)",
            "  * No inline environment variables (VAR=value cmd)",
            "  * No cd commands (commands run from repo root)",
            "  * Each command must be a single executable with arguments only.",
        ]
    )


def _execute_tool(sb: Sandbox, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    """Execute a sandbox tool by name with the provided arguments.

    Note: sandbox.run is intentionally NOT exposed to the model.
    The controller handles test execution directly for security.
    """
    if not isinstance(args, dict):
        args = {}
    if tool == "sandbox.clone_repo":
        return clone_public_github(sb, args.get("github_url", ""))
    if tool == "sandbox.checkout":
        return checkout(sb, args.get("ref", ""))
    # sandbox.run intentionally removed - controller-only for security
    if tool == "sandbox.read_file":
        try:
            max_bytes = int(args.get("max_bytes", 120000))
        except (ValueError, TypeError):
            max_bytes = 120000
        return read_file(sb, args.get("path", ""), max_bytes=max_bytes)
    if tool == "sandbox.grep":
        try:
            max_matches = int(args.get("max_matches", 200))
        except (ValueError, TypeError):
            max_matches = 200
        return grep(sb, args.get("query", ""), max_matches=max_matches)
    if tool == "sandbox.list_tree":
        try:
            max_files = int(args.get("max_files", 400))
        except (ValueError, TypeError):
            max_files = 400
        return list_tree(sb, max_files=max_files)
    if tool == "sandbox.apply_patch":
        return apply_patch(sb, args.get("diff", ""))
    if tool == "sandbox.git_status":
        return git_status(sb)
    if tool == "sandbox.reset_hard":
        return reset_hard(sb)
    if tool == "sandbox.pip_install":
        try:
            timeout = int(args.get("timeout_sec", 300))
        except (ValueError, TypeError):
            timeout = 300
        return pip_install(sb, args.get("packages", ""), timeout_sec=timeout)
    if tool == "sandbox.pip_install_requirements":
        try:
            timeout = int(args.get("timeout_sec", 300))
        except (ValueError, TypeError):
            timeout = 300
        return pip_install_requirements(
            sb, args.get("requirements_file", "requirements.txt"), timeout_sec=timeout
        )
    if tool == "sandbox.create_venv":
        try:
            timeout = int(args.get("timeout_sec", 60))
        except (ValueError, TypeError):
            timeout = 60
        return create_venv(sb, args.get("venv_path", ".venv"), timeout_sec=timeout)
    if tool == "sandbox.pip_install_progressive":
        try:
            timeout = int(args.get("timeout_sec", 300))
        except (ValueError, TypeError):
            timeout = 300
        return pip_install_progressive(sb, args.get("packages", ""), timeout_sec=timeout)
    if tool == "sandbox.find_local_module":
        return find_local_module(sb, args.get("module_name", ""))
    if tool == "sandbox.set_pythonpath":
        return set_pythonpath(sb, args.get("path", ""))
    if tool == "sandbox.run_command":
        try:
            timeout = int(args.get("timeout_sec", 120))
        except (ValueError, TypeError):
            timeout = 120
        # Check command allowlist internally in docker_run/runner if needed,
        # but here we allow general execution subject to container security.
        # Commands are run as the sandbox user.
        cmd = args.get("command", "")
        if isinstance(cmd, list):
            cmd = " ".join(str(c) for c in cmd)
        res = docker_run(sb, str(cmd), timeout_sec=timeout)
        return {
            "ok": res.ok,
            "exit_code": res.exit_code,
            "stdout": res.stdout,
            "stderr": res.stderr,
            "timed_out": res.timed_out,
        }
    return {"ok": False, "error": f"Unknown tool: {tool}"}


def _collect_relevant_files(sb: Sandbox, v: VerifyResult, repo_tree: str) -> list[dict[str, Any]]:
    """Collect a small set of files likely related to the failure.

    The selection includes the first failing test file and any Python files
    mentioned in tracebacks. File paths are normalized and filtered via
    _safe_path to avoid sending forbidden files to the model.
    """
    out: list[dict[str, Any]] = []
    # failing test file
    if v.failing_tests:
        tp = normalize_test_path(v.failing_tests[0])
        if _safe_path(tp):
            out.append(read_file(sb, tp, max_bytes=120000))
    # traceback referenced files
    combined = (v.stdout or "") + "\n" + (v.stderr or "")
    for p in parse_trace_files(combined, limit=6):
        # trace files may be absolute; ignore abs outside
        p2 = p.replace("\\", "/")
        if p2.startswith(sb.repo_dir.replace("\\", "/")):
            p2 = p2[len(sb.repo_dir) :].lstrip("/")
        if p2.endswith(".py") and _safe_path(p2):
            out.append(read_file(sb, p2, max_bytes=120000))
    return out


def _collect_relevant_files_quixbugs(
    sb: Sandbox, v: VerifyResult, repo_tree: str
) -> list[dict[str, Any]]:
    """Collect files for QuixBugs repositories with specific heuristics.

    QuixBugs structure:
    - python_testcases/test_<program>.py (test files)
    - python_programs/<program>.py (implementation files)

    Strategy:
    1. Always include the first failing test file (highest priority)
    2. Map test file to corresponding program file
    3. Include any traceback-referenced files
    4. Add common helper files if referenced
    """
    out: list[dict[str, Any]] = []

    if not v.failing_tests:
        return out

    # Get the first failing test file (highest priority)
    test_path = normalize_test_path(v.failing_tests[0])
    if not _safe_path(test_path):
        return out

    # 1. Include the failing test file (highest priority)
    test_content = read_file(sb, test_path, max_bytes=120000)
    if test_content.get("ok"):
        out.append(test_content)

    # 2. Map test file to program file
    # python_testcases/test_quicksort.py -> python_programs/quicksort.py
    if "python_testcases/" in test_path:
        test_filename = test_path.split("/")[-1]  # test_quicksort.py
        if test_filename.startswith("test_") and test_filename.endswith(".py"):
            program_name = test_filename[5:]  # quicksort.py (remove "test_")
            program_path = f"python_programs/{program_name}"
            if _safe_path(program_path):
                program_content = read_file(sb, program_path, max_bytes=120000)
                if program_content.get("ok"):
                    out.append(program_content)

    # 3. Include traceback-referenced files
    combined = (v.stdout or "") + "\n" + (v.stderr or "")
    for p in parse_trace_files(combined, limit=6):
        p2 = p.replace("\\", "/")
        if p2.startswith(sb.repo_dir.replace("\\", "/")):
            p2 = p2[len(sb.repo_dir) :].lstrip("/")
        if p2.endswith(".py") and _safe_path(p2):
            # Avoid duplicates
            if not any(f.get("path") == p2 for f in out):
                file_content = read_file(sb, p2, max_bytes=120000)
                if file_content.get("ok"):
                    out.append(file_content)

    return out


def _evaluate_patch_in_worktree(
    sb: Sandbox, diff: str, focus_cmd: str, full_cmd: str
) -> tuple[bool, str]:
    """Test a candidate patch in a detached worktree before applying to main repo."""
    wt = make_worktree(sb)
    try:
        ap = apply_patch_in_dir(wt, diff)
        if not ap.get("ok"):
            return False, f"apply_failed: {ap.get('stderr', '')}{ap.get('stdout', '')}"
        r1 = run_cmd(Sandbox(sb.root, wt), focus_cmd, timeout_sec=90)
        if not r1.get("ok"):
            return False, "focus_failed:\n" + (r1.get("stdout", "") + r1.get("stderr", ""))
        r2 = run_cmd(Sandbox(sb.root, wt), full_cmd, timeout_sec=180)
        if r2.get("ok"):
            return True, "PASS"
        return False, "full_failed:\n" + (r2.get("stdout", "") + r2.get("stderr", ""))
    except Exception as e:
        return False, f"exception: {type(e).__name__}: {e!s}"
    finally:
        try:
            drop_worktree(sb, wt)
        except Exception:
            pass


@dataclass
class BudgetConfig:
    """Budget configuration for resource limits."""
    
    max_steps: int = 0
    max_llm_calls: int = 0
    max_tokens: int = 0
    max_time_seconds: float = 0
    max_subprocess_calls: int = 0
    warning_threshold: float = 0.8


@dataclass
class ContractsConfig:
    """Contracts configuration for runtime safety checks."""
    
    enabled: bool = True
    shell_execution_enabled: bool = True
    budget_tracking_enabled: bool = True
    llm_calling_enabled: bool = True
    event_logging_enabled: bool = True


@dataclass
class ControllerConfig:
    """Configuration for a controller run."""

    github_url: str
    test_cmd: str = "pytest -q"
    ref: str | None = None
    max_steps: int = 12
    temps: list[float] = field(default_factory=lambda: [0.0, 0.2, 0.4])
    fix_all: bool = False
    max_steps_without_progress: int = 10
    collect_finetuning_data: bool = False
    model: str = "deepseek-chat"
    max_minutes: int = 30
    install_timeout: int = 300
    focus_timeout: int = 120
    full_timeout: int = 300
    max_tool_calls: int = 40
    docker_image: str = "python:3.11-slim"
    unsafe_host_exec: bool = False
    cpu: float = 2.0
    mem_mb: int = 4096
    pids: int = 256
    docker_readonly: bool = False
    lint_cmd: str | None = None
    typecheck_cmd: str | None = None
    repro_cmd: str | None = None
    verify_cmd: str | None = None
    dry_run: bool = False
    project_type: str = "auto"
    buildpack: str = "auto"
    enable_sysdeps: bool = False
    sysdeps_tier: int = 4
    sysdeps_max_packages: int = 10
    build_cmd: str | None = None
    learning_db_path: str | None = None
    learning_half_life_days: int = 14
    learning_max_age_days: int = 90
    learning_max_rows: int = 20000
    time_mode: str = "frozen"  # frozen|live
    run_started_at_utc: str | None = None
    time_seed: int | None = None
    rng_seed: int | None = None
    feature_mode: bool = False
    feature_description: str | None = None
    acceptance_criteria: list[str] = field(default_factory=list)
    # Verification configuration for feature mode
    verify_policy: str = "tests_only"  # tests_only | cmds_then_tests | cmds_only
    focused_verify_cmds: list[str] = field(default_factory=list)
    verify_cmds: list[str] = field(default_factory=list)
    # Hygiene configuration overrides
    max_lines_changed: int | None = None
    max_files_changed: int | None = None
    allow_lockfile_changes: bool = False
    # Phase budget limits for reliability
    max_install_attempts: int = 3
    max_patch_attempts: int = 20
    max_verification_attempts: int = 5
    # Verification repeatability
    repro_times: int = 1  # Run verification N times to ensure reproducibility
    # Performance optimizations
    enable_llm_cache: bool = False  # Enable LLM response caching
    llm_cache_path: str | None = None  # Path to LLM cache database
    parallel_patches: bool = True  # Generate patches in parallel (faster)
    ensemble_mode: bool = False  # Use multi-model ensemble
    incremental_tests: bool = False  # Run only affected tests first
    enable_telemetry: bool = False  # Enable OpenTelemetry/Prometheus
    telemetry_port: int = 8080  # Prometheus metrics port
    # Elite Controller options
    policy_mode: str = "off"  # off | bandit
    planner_mode: str = "off"  # off | dag | v2 | v5
    repo_index: bool = False  # Enable repo indexing
    seed: int = 1337  # Deterministic seed
    # Risk & persistence
    risk_profile: str = "production"  # production | research
    state_dir: str | None = None  # base host dir; we create <base>/<risk>/<run_id>/
    # Verification durability
    durability_reruns: int = 0  # rerun full tests N additional times after success
    no_eval: bool = False  # Skip final evaluation
    # Context-related configuration (for create_context compatibility)
    output_dir: str = ".rfsn"  # Output directory for artifacts
    events_file: str = "events.jsonl"  # Events log filename
    plan_file: str = "plan.json"  # Plan filename
    # Budget configuration (inline for context compatibility)
    budget: BudgetConfig = field(default_factory=lambda: BudgetConfig())
    # Contracts configuration (inline for context compatibility)
    contracts: ContractsConfig = field(default_factory=lambda: ContractsConfig())
    # Beam search configuration
    beam_search_enabled: bool = False  # Enable multi-step beam search
    beam_width: int = 3  # Number of candidates to keep per step
    beam_depth: int = 5  # Maximum expansion depth
    beam_score_threshold: float = 0.95  # Score to terminate early
    beam_timeout_seconds: float = 300.0  # Total search timeout


def run_controller(cfg: ControllerConfig) -> dict[str, Any]:
    """Run the controller loop until the goal is reached or max_steps exhausted.

    Args:
        cfg: The controller configuration.

    Returns:
        A dictionary indicating success, or error details, and where the
        sandbox directory can be inspected.
    """
    start_dt = None
    if cfg.run_started_at_utc:
        start_dt = parse_utc_iso(cfg.run_started_at_utc)
    if start_dt is None:
        start_dt = SystemClock().now_utc()

    if (cfg.time_mode or "").lower() == "live":
        clock = SystemClock()
    else:
        clock = FrozenClock(start_dt)  # type: ignore

    seed_material = {
        "repo": cfg.github_url,
        "ref": cfg.ref,
        "test_cmd": cfg.test_cmd,
        "model": cfg.model,
        "run_started_at_utc": start_dt.astimezone(UTC).isoformat(),
        "time_mode": (cfg.time_mode or "").lower() or "frozen",
    }
    if cfg.time_seed is None:
        cfg.time_seed = int(hashlib.sha256(str(seed_material).encode("utf-8")).hexdigest()[:8], 16)
    if cfg.rng_seed is None:
        cfg.rng_seed = int(
            hashlib.sha256(f"rng:{cfg.time_seed}".encode()).hexdigest()[:8], 16
        )

    seed_material["time_seed"] = str(int(cfg.time_seed) if cfg.time_seed is not None else 0)
    seed_material["rng_seed"] = str(int(cfg.rng_seed) if cfg.rng_seed is not None else 0)
    run_id = make_run_id(clock=clock, seed_material=seed_material)

    random.seed(int(cfg.rng_seed))
    try:
        import numpy as np  # type: ignore

        np.random.seed(int(cfg.rng_seed))
    except Exception:
        pass

    # Initialize variables that will be used in exception handler
    sb = None
    log_dir = None
    evidence_exporter = None
    command_log: list[dict[str, Any]] = []
    memory_store: ActionOutcomeStore | None = None

    try:
        sb = create_sandbox(run_id=run_id)
        # Determine log directory based on state_dir and risk_profile
        if cfg.state_dir:
            base = Path(cfg.state_dir).expanduser().resolve()
            log_dir = str(base / cfg.risk_profile / run_id)
            os.makedirs(log_dir, exist_ok=True)
        else:
            log_dir = sb.root  # write logs next to sandbox for inspection
        # Create evidence exporter with output_dir inside log_dir/results
        evidence_exporter = EvidencePackExporter(
            EvidencePackConfig(output_dir=os.path.join(log_dir, "results"))
        )

        def log(rec: dict[str, Any]) -> None:
            write_jsonl(log_dir, rec, clock=clock)

        # Initialize contract system for operation validation
        try:
            from .contracts import (
                get_global_registry,
                register_standard_contracts,
            )
            registry = get_global_registry()
            # Register standard contracts if not already registered
            if not registry.has_contract("shell_execution"):
                register_standard_contracts(registry)
        except ImportError:
            pass  # Contracts module not available

        bad_hashes: set[str] = set()
        observations: str = ""  # buffer for tool results to feed back to model
        patch_attempts: int = 0  # count patch attempts to detect lack of progress
        steps_without_progress: int = 0  # track steps without reducing failing tests
        min_failing_tests: int = 999999  # track minimum failing tests seen
        distinct_sigs: set[str] = set()  # track distinct error signatures for multi-bug detection
        bailout_reason: str | None = None
        low_conf_streak: int = 0
        
        # Initialize optimization components
        termination_heuristics = None
        try:
            from .optimizations import TerminationHeuristics
            termination_heuristics = TerminationHeuristics()
        except ImportError:
            pass
        
        # Initialize adaptive patch budget controller
        # Determines max_lines/max_files dynamically based on stagnation
        patch_budget = create_patch_budget_controller(
            user_ceiling_override=getattr(cfg, "allow_ceiling_override", False),
            stagnation_threshold=2,
        )
        log({"phase": "patch_budget_init", **patch_budget.get_state_summary()})
        
        # Initialize diff minimizer for pre-hygiene patch shrinking
        diff_minimizer = DiffMinimizer()
        
        # Delta tracker initialized after first test run (needs baseline)
        delta_tracker: TestDeltaTracker | None = None
        
        # Initialize QA orchestrator for claim-based verification
        qa_orchestrator: QAOrchestrator | None = None
        if cfg.learning_db_path:
            qa_db_path = cfg.learning_db_path.replace(".db", "_qa.db")
        else:
            qa_db_path = None
        budget_limits = patch_budget.get_limits()
        qa_config = QAConfig(
            surgical_max_lines=budget_limits[0],
            surgical_max_files=budget_limits[1],
            persist_outcomes=qa_db_path is not None,
            db_path=qa_db_path,
        )
        qa_orchestrator = QAOrchestrator(
            config=qa_config,
            delta_tracker=delta_tracker,
        )
        log({"phase": "qa_orchestrator_init", "db_path": qa_db_path})

        
        # Initialize telemetry if enabled
        if cfg.enable_telemetry:
            try:
                from .telemetry import init_telemetry
                init_telemetry(
                    enabled=True,
                    enable_metrics=True,
                    metrics_port=cfg.telemetry_port,
                )
                log({"phase": "telemetry_init", "port": cfg.telemetry_port})
            except ImportError:
                log({"phase": "telemetry_init", "error": "telemetry module not available"})

        # Initialize vNext components
        current_phase = Phase.INGEST
        tool_manager = ToolRequestManager(
            ToolRequestConfig(max_total_requests_per_run=cfg.max_tool_calls)
        )
        stall_state = StallState()

        # Track baseline for evidence pack
        baseline_output = ""
        final_output = ""
        final_output = ""
        winner_diff: str | None = None
        feature_summary = None  # Store feature summary for evidence pack
        feature_summary = None  # Store feature summary for evidence pack
        log(
            {
                "phase": "run_header",
                "run_id": run_id,
                "run_started_at_utc": start_dt.astimezone(UTC).isoformat(),
                "time_seed": int(cfg.time_seed or 0),
                "time_mode": (cfg.time_mode or "").lower() or "frozen",
                "rng_seed": int(cfg.rng_seed or 0),
            }
        )
        log({"phase": "init", "cfg": cfg.__dict__})

        # Initialize Broadcaster
        broadcaster = ProgressBroadcaster(run_id=run_id)
        broadcaster.log(f"Run {run_id} started", level="info")
        broadcaster.status("INGEST")

        if cfg.learning_db_path:
            db_path = os.path.expanduser(cfg.learning_db_path)
            if not os.path.isabs(db_path):
                db_path = os.path.abspath(db_path)
            memory_store = ActionOutcomeStore(
                db_path,
                half_life_days=cfg.learning_half_life_days,
                max_age_days=cfg.learning_max_age_days,
                max_rows=cfg.learning_max_rows,
            )
            log(
                {
                    "phase": "learning_init",
                    "db_path": db_path,
                    "half_life_days": cfg.learning_half_life_days,
                    "max_age_days": cfg.learning_max_age_days,
                    "max_rows": cfg.learning_max_rows,
                }
            )

        # === ELITE CONTROLLER: Planner/Policy flags for later init ===
        elite_planner = None  # Will be initialized after context is set up
        elite_planner_enabled = cfg.planner_mode == "dag"
        if elite_planner_enabled:
            log({"phase": "elite_planner_init", "mode": cfg.planner_mode, "status": "deferred"})

        # === PLANNER V2: High-level goal decomposition ===
        planner_v2_adapter = None
        planner_v2_enabled = cfg.planner_mode == "v2"
        if planner_v2_enabled:
            try:
                from .planner_v2 import ControllerAdapter, MemoryAdapter, PlannerV2
                planner_memory = MemoryAdapter(memory_store) if memory_store else MemoryAdapter()
                planner_v2 = PlannerV2(
                    memory_adapter=planner_memory,
                    seed=cfg.seed,
                    state_dir=Path(log_dir),
                )
                planner_v2_adapter = ControllerAdapter(planner=planner_v2)
                log({"phase": "planner_v2_init", "mode": cfg.planner_mode, "status": "ready"})
            except ImportError as e:
                log({"phase": "planner_v2_init", "error": str(e)})
                planner_v2_enabled = False

        # === PLANNER V5: Meta-planning with state tracking ===
        planner_v5_adapter = None
        planner_v5_enabled = cfg.planner_mode == "v5"
        if planner_v5_enabled:
            try:
                from .planner_v5_adapter import PlannerV5Adapter
                planner_v5_adapter = PlannerV5Adapter(enabled=True)
                if planner_v5_adapter.enabled:
                    log({"phase": "planner_v5_init", "mode": cfg.planner_mode, "status": "ready"})
                else:
                    log({"phase": "planner_v5_init", "mode": cfg.planner_mode, "status": "unavailable"})
                    planner_v5_enabled = False
            except ImportError as e:
                log({"phase": "planner_v5_init", "error": str(e)})
                planner_v5_enabled = False

        # === ELITE CONTROLLER: Initialize Policy ===
        elite_policy = None
        if cfg.policy_mode == "bandit":
            try:
                from .policy_bandit import create_policy
                policy_db = cfg.learning_db_path.replace(".db", "_policy.db") if cfg.learning_db_path else None
                elite_policy = create_policy(db_path=policy_db, seed=cfg.seed)
                log({"phase": "elite_policy_init", "mode": cfg.policy_mode, "seed": cfg.seed})
            except ImportError as e:
                log({"phase": "elite_policy_init", "error": str(e)})

        # === ELITE CONTROLLER: Initialize Repo Index ===
        elite_repo_index = None
        if cfg.repo_index:
            try:
                from .repo_index import RepoIndex
                elite_repo_index = RepoIndex.build(sb.repo_dir)
                log({"phase": "elite_repo_index_init", "files": len(elite_repo_index.files)})
            except ImportError as e:
                log({"phase": "elite_repo_index_init", "error": str(e)})

        # === PHASE: INGEST ===
        log(PhaseTransition(None, Phase.INGEST).to_dict())

        # Validate GitHub URL
        is_valid, normalized_url, url_error = validate_github_url(cfg.github_url)
        if not is_valid:
            return {"ok": False, "error": f"Invalid GitHub URL: {url_error}"}

        github_url = normalized_url
        log({"phase": "url_validation", "normalized_url": github_url})

        # Clone repository
        r = clone_public_github(sb, github_url or "")
        log({"phase": "clone", "result": r})
        if not r.get("ok"):
            return {"ok": False, "error": r.get("error") or r.get("stderr")}

        # Checkout ref if specified
        if cfg.ref:
            co = checkout(sb, cfg.ref)
            log({"phase": "checkout", "result": co})
            if not co.get("ok"):
                return {"ok": False, "error": co.get("stderr")}

        reset_hard(sb)
        tree = list_tree(sb, max_files=2000)
        repo_tree = tree.get("files", []) if tree.get("ok") else []
        repo_tree_text = "\n".join(repo_tree)

        # === PHASE: DETECT ===
        current_phase = Phase.DETECT
        log(PhaseTransition(Phase.INGEST, Phase.DETECT).to_dict())

        # === PHASE: V3 BUILDPACK DETECTION ===
        selected_buildpack = None
        selected_buildpack_instance = None
        try:
            # Create buildpack context
            buildpack_ctx = BuildpackContext(
                repo_dir=sb.repo_dir,
                repo_tree=repo_tree,
                files={},
            )

            # Read relevant files for buildpack detection
            buildpack_files = [
                "pyproject.toml",
                "requirements.txt",
                "setup.py",
                "setup.cfg",
                "Pipfile",
                "poetry.lock",
                "conftest.py",
                "pytest.ini",
                "py.typed",  # Python indicators
                "package.json",
                "package-lock.json",
                "yarn.lock",
                "pnpm-lock.yaml",
                "bun.lockb",
                "go.mod",
                "go.sum",
                "Cargo.toml",
                "Cargo.lock",
                "pom.xml",
                "build.gradle",
                "build.gradle.kts",
                "gradlew",
                "global.json",
            ]

            for filename in buildpack_files:
                match_path = next(
                    (f for f in repo_tree if f == filename),
                    None,
                )
                if not match_path:
                    continue
                try:
                    rf = read_file(sb, match_path)
                    if isinstance(rf, dict) and rf.get("ok") and rf.get("content"):
                        buildpack_ctx.files[filename] = rf["content"]
                except Exception:
                    pass

            # Detect buildpack
            all_buildpacks = get_all_buildpacks()
            best_result = None
            best_buildpack = None

            for buildpack in all_buildpacks:
                result = buildpack.detect(buildpack_ctx)
                if result and result.confidence > 0.5:
                    if best_result is None or result.confidence > best_result.confidence:
                        best_result = result
                        best_buildpack = buildpack

            if best_buildpack and best_result:
                selected_buildpack_instance = best_buildpack
                selected_buildpack = best_buildpack.image()
                log(
                    {
                        "phase": "buildpack_detect",
                        "buildpack_type": best_result.buildpack_type.value,
                        "confidence": best_result.confidence,
                        "image": selected_buildpack,
                        "metadata": best_result.metadata,
                    }
                )
            else:
                # Fallback to docker_image
                selected_buildpack = cfg.docker_image
                log(
                    {
                        "phase": "buildpack_detect",
                        "error": "No buildpack detected",
                        "fallback_image": selected_buildpack,
                    }
                )

            # Override buildpack if --project-type is explicitly set (not "auto")
            if cfg.project_type and cfg.project_type != "auto":
                type_map = {
                    "python": BuildpackType.PYTHON,
                    "node": BuildpackType.NODE,
                    "go": BuildpackType.GO,
                    "rust": BuildpackType.RUST,
                    "java": BuildpackType.JAVA,
                    "dotnet": BuildpackType.DOTNET,
                }
                if cfg.project_type in type_map:
                    forced = get_buildpack(type_map[cfg.project_type])
                    selected_buildpack_instance = forced
                    selected_buildpack = forced.image()
                    log(
                        {
                            "phase": "buildpack_override",
                            "reason": "project_type_flag",
                            "buildpack_type": cfg.project_type,
                            "image": selected_buildpack,
                        }
                    )

            if cfg.test_cmd and cfg.test_cmd != "pytest -q":
                inferred = _infer_buildpack_type_from_test_cmd(cfg.test_cmd)
                if inferred is not None:
                    forced = get_buildpack(inferred)
                    selected_buildpack_instance = forced
                    selected_buildpack = forced.image()
                    log(
                        {
                            "phase": "buildpack_override",
                            "reason": "test_cmd",
                            "buildpack_type": inferred.value,
                            "image": selected_buildpack,
                        }
                    )
        except Exception as e:
            # Fallback to docker_image if buildpack detection fails
            selected_buildpack = cfg.docker_image
            log({"phase": "buildpack_detect", "error": str(e)})

        # Legacy detection for backward compatibility
        project_type = detect_project_type(sb.repo_dir)
        setup_commands = get_setup_commands(sb.repo_dir)
        detected_test_cmd = get_default_test_command(sb.repo_dir)

        # Use detected test command if not overridden
        effective_test_cmd = (
            cfg.test_cmd if cfg.test_cmd != "pytest -q" else (detected_test_cmd or "pytest -q")
        )

        log(
            {
                "phase": "detect",
                "project_type": project_type.name if project_type else None,
                "setup_commands": setup_commands,
                "test_cmd": effective_test_cmd,
                "buildpack_image": selected_buildpack,
            }
        )

        # === WIRE LANGUAGE-SCOPED ALLOWLIST ===
        # Set sandbox allowed_commands based on detected project type
        from .allowlist_profiles import commands_for_project

        # Build project info dict for allowlist selection, using the buildpack-derived language
        project_info = {
            "language": selected_buildpack_instance.buildpack_type.value
            if selected_buildpack_instance
            else None,
        }
        sb.allowed_commands = commands_for_project(project_info)
        log(
            {
                "phase": "allowlist_configured",
                "language": project_info["language"],
                "num_commands": len(sb.allowed_commands),
            }
        )

        # Detect QuixBugs repository structure
        is_quixbugs = "python_testcases/" in repo_tree_text and "python_programs/" in repo_tree_text

        # === PHASE: SETUP ===
        current_phase = Phase.SETUP
        log(PhaseTransition(Phase.DETECT, Phase.SETUP).to_dict())

        # Track setup results for validation
        setup_results = {}
        lockfile_path = None

        # Use buildpack install plan if available
        if selected_buildpack_instance:
            print(f"[SETUP] Using buildpack: {selected_buildpack_instance.buildpack_type.value}")
            install_steps = selected_buildpack_instance.install_plan(buildpack_ctx)

            setup_key_map = {
                "python": "pip",
                "node": "node",
                "go": "go",
                "rust": "rust",
                "java": "java",
                "dotnet": "dotnet",
            }
            setup_key = setup_key_map.get(selected_buildpack_instance.buildpack_type.value)

            for install_step in install_steps:
                print(f"[SETUP] Running: {install_step.description}")
                cmd_str = " ".join(install_step.argv)
                if cfg.unsafe_host_exec:
                    r = run_cmd(sb, cmd_str, timeout_sec=install_step.timeout_sec)
                    install_result = DockerResult(
                        ok=bool(r.get("ok")),
                        exit_code=int(r.get("exit_code") or 1),
                        stdout=r.get("stdout") or "",
                        stderr=r.get("stderr") or "",
                        timed_out=False,
                    )
                else:
                    install_result = docker_install(
                        sb, cmd_str, timeout_sec=install_step.timeout_sec, docker_image=selected_buildpack
                    )

                # Store result by step description
                setup_results[install_step.description] = install_result
                if setup_key:
                    prev = setup_results.get(setup_key)
                    if prev is None or prev.ok:
                        setup_results[setup_key] = install_result

                command_log.append(
                    {
                        "phase": "setup",
                        "command": cmd_str,
                        "exit_code": install_result.exit_code,
                        "ok": install_result.ok,
                        "stdout": install_result.stdout[:1000],
                        "stderr": install_result.stderr[:1000],
                    }
                )
                log(
                    {
                        "phase": "setup",
                        "command": cmd_str,
                        "result": {"ok": install_result.ok, "exit_code": install_result.exit_code},
                        "stdout": install_result.stdout[:1000],
                        "stderr": install_result.stderr[:1000],
                    }
                )
                if not install_result.ok:
                    print(f"[SETUP] Failed: {install_result.stderr[:200]}")
        elif setup_commands:
            for setup_cmd in setup_commands:
                print(f"[SETUP] Running: {setup_cmd}")
                if cfg.unsafe_host_exec:
                    r = run_cmd(sb, setup_cmd, timeout_sec=cfg.install_timeout)
                    install_result = DockerResult(
                        ok=bool(r.get("ok")),
                        exit_code=int(r.get("exit_code") or 1),
                        stdout=r.get("stdout") or "",
                        stderr=r.get("stderr") or "",
                        timed_out=False,
                    )
                else:
                    install_result = docker_install(
                        sb,
                        setup_cmd,
                        timeout_sec=cfg.install_timeout,
                        docker_image=selected_buildpack,
                    )

                # Store result by command type
                cmd_lower = setup_cmd.lower()
                if "pip install" in cmd_lower or "python -m pip" in cmd_lower:
                    setup_results["pip"] = install_result
                elif "npm install" in cmd_lower or "npm ci" in cmd_lower:
                    setup_results["node"] = install_result
                elif "go mod" in cmd_lower:
                    setup_results["go"] = install_result
                elif "cargo" in cmd_lower:
                    setup_results["rust"] = install_result
                elif "mvn" in cmd_lower or "gradle" in cmd_lower:
                    setup_results["java"] = install_result
                elif "dotnet restore" in cmd_lower:
                    setup_results["dotnet"] = install_result

                command_log.append(
                    {
                        "phase": "setup",
                        "command": setup_cmd,
                        "exit_code": install_result.exit_code,
                        "ok": install_result.ok,
                        "stdout": install_result.stdout[:1000],
                        "stderr": install_result.stderr[:1000],
                    }
                )
                log(
                    {
                        "phase": "setup",
                        "command": setup_cmd,
                        "result": {"ok": install_result.ok, "exit_code": install_result.exit_code},
                        "stdout": install_result.stdout[:1000],
                        "stderr": install_result.stderr[:1000],
                    }
                )
                if not install_result.ok:
                    print(f"[SETUP] Failed: {install_result.stderr[:200]}")

        # Detect lockfile
        if selected_buildpack_instance:
            # Lockfile detection is handled by buildpack metadata
            if best_result and best_result.metadata:
                lockfile_path = best_result.metadata.get("lockfile")

        # === PHASE: SYSDEPS (V3) ===
        sysdeps_installed = []
        sysdeps_blocked = []

        if cfg.enable_sysdeps and selected_buildpack_instance:
            current_phase = Phase.SETUP
            log({"phase": "sysdeps", "enabled": True})

            # Use buildpack's sysdeps whitelist
            sysdeps_whitelist = selected_buildpack_instance.sysdeps_whitelist()

            tier_map = {
                0: AptTier.TIER_0,
                1: AptTier.TIER_1,
                2: AptTier.TIER_2,
                3: AptTier.TIER_3,
                4: AptTier.TIER_4,
                5: AptTier.TIER_5,
                6: AptTier.TIER_6,
                7: AptTier.TIER_7,
            }

            whitelist = AptWhitelist(
                max_packages=cfg.sysdeps_max_packages,
                max_tier=tier_map.get(cfg.sysdeps_tier, AptTier.TIER_4),
                allow_wildcards=False,
                custom_packages=sysdeps_whitelist,
            )

            installer = SysdepsInstaller(
                whitelist=whitelist,
                dry_run=False,
            )

            print("[SYSDEPS] Installing system dependencies...")
            sysdeps_result = installer.install(
                packages=[],
                hints=[],
            )

            sysdeps_installed = sysdeps_result.installed_packages
            sysdeps_blocked = sysdeps_result.blocked_packages

            log(
                {
                    "phase": "sysdeps",
                    "result": {
                        "success": sysdeps_result.success,
                        "installed_packages": sysdeps_installed,
                        "blocked_packages": sysdeps_blocked,
                        # "attempted": sysdeps_result.attempted_packages,
                    },
                }
            )

            if sysdeps_result.success:
                print(f"[SYSDEPS] Installed: {sysdeps_installed}")
            else:
                print(f"[SYSDEPS] Failed: {sysdeps_result.error_message}")
                if sysdeps_blocked:
                    print(f"[SYSDEPS] Blocked packages: {sysdeps_blocked}")

        # === PHASE: SETUP VALIDATION ===
        # Create setup report and check if we should bail out
        setup_report = create_setup_report(
            pip_result=setup_results.get("pip"),
            node_result=setup_results.get("node"),
            go_result=setup_results.get("go"),
            rust_result=setup_results.get("rust"),
            java_result=setup_results.get("java"),
            dotnet_result=setup_results.get("dotnet"),
            lockfile_path=lockfile_path,
            sysdeps_installed=sysdeps_installed,
            sysdeps_failed=[],
            sysdeps_blocked=sysdeps_blocked,
        )

        log(
            {
                "phase": "setup_validation",
                "report": setup_report.to_dict(),
            }
        )

        # Hard bailout if setup failed
        if setup_report.should_bailout():
            bailout_reason = setup_report.get_bailout_message()
            print(f"\n[BAILOUT] {bailout_reason}")
            log(
                {
                    "phase": "bailout",
                    "reason": bailout_reason,
                    "setup_report": setup_report.to_dict(),
                }
            )

            return {
                "ok": False,
                "error": bailout_reason,
                "sandbox": sb.root,
                "repo_dir": sb.repo_dir,
                "phase": "setup_failed",
            }

        print("\n[SETUP_VALIDATION] Setup passed")
        if setup_report.has_lockfile:
            print(f"  Lockfile found: {setup_report.lockfile_path}")
        if setup_report.sysdeps_installed:
            print(f"  System deps installed: {setup_report.sysdeps_installed}")

        # === PHASE: BASELINE ===
        current_phase = Phase.BASELINE
        log(PhaseTransition(Phase.SETUP, Phase.BASELINE).to_dict())

        # Use user-provided test command if available, otherwise use buildpack test plan
        if cfg.test_cmd and cfg.test_cmd != "pytest -q":
            # User explicitly provided a test command
            effective_test_cmd = cfg.test_cmd
        elif selected_buildpack_instance:
            # Use buildpack test plan
            test_plan = selected_buildpack_instance.test_plan(buildpack_ctx)
            effective_test_cmd = " ".join(test_plan.argv)
        else:
            # Use detected test command
            effective_test_cmd = detected_test_cmd or "pytest -q"

        # Run baseline tests
        print(f"\n[BASELINE] Running: {effective_test_cmd}")
        v = _run_tests_in_sandbox(
            sb,
            effective_test_cmd,
            cfg,
            command_log,
            selected_buildpack,
            selected_buildpack_instance,
        )
        baseline_output = (v.stdout or "") + "\n" + (v.stderr or "")

        log(
            {
                "phase": "baseline",
                "tests_ok": v.ok,
                "exit_code": v.exit_code,
                "failing_tests": v.failing_tests[:10],
                "sig": v.sig,
            }
        )

        # Handle pytest exit code 2 (no tests found)
        if v.exit_code == 2 and not v.ok:
            print("\n[BASELINE] Exit code 2 detected - no tests found")
            # Try alternative test commands based on buildpack
            if selected_buildpack_instance:
                # Buildpack-specific fallbacks
                if selected_buildpack_instance.buildpack_type.value == "python":
                    suggestions = [
                        "python -m pytest -q --collect-only",
                        "python -m pytest -q tests/",
                        "python -m unittest discover -q",
                    ]
                elif selected_buildpack_instance.buildpack_type.value == "node":
                    suggestions = [
                        "npm test -- --listTests",
                        "npm test -- tests/",
                    ]
                else:
                    suggestions = []

                if suggestions:
                    suggested_cmd = suggestions[0]
                    print(f"\n[BASELINE] Retrying with: {suggested_cmd}")
                    v = _run_tests_in_sandbox(
                        sb,
                        suggested_cmd,
                        cfg,
                        command_log,
                        selected_buildpack,
                        selected_buildpack_instance,
                    )
                    baseline_output = (v.stdout or "") + "\n" + (v.stderr or "")

                    log(
                        {
                            "phase": "baseline_retry",
                            "tests_ok": v.ok,
                            "exit_code": v.exit_code,
                            "failing_tests": v.failing_tests[:10],
                            "sig": v.sig,
                            "test_cmd": suggested_cmd,
                        }
                    )
                    if v.ok or v.exit_code != 2:
                        effective_test_cmd = suggested_cmd
                        print(f"  Using new test command: {effective_test_cmd}")

        if v.ok:
            print("\n[BASELINE] SUCCESS! All tests passing at baseline.")
            return {
                "ok": True,
                "sandbox": sb.root,
                "repo_dir": sb.repo_dir,
                "steps_taken": 0,
                "phase": "baseline_pass",
            }

        # === PHASE: REPAIR_LOOP ===
        current_phase = Phase.REPAIR_LOOP
        log(PhaseTransition(Phase.BASELINE, Phase.REPAIR_LOOP).to_dict())

        # If fix_all mode, use unlimited steps
        max_iterations = float("inf") if cfg.fix_all else cfg.max_steps
        step_count = 0

        # Budget tracking for reliability
        total_tool_calls = 0
        total_patch_attempts = 0
        total_verification_attempts = 0

        # Feature mode tracking
        feature_subgoals = list(DEFAULT_FEATURE_SUBGOALS)
        completed_feature_subgoals: list[str] = []
        current_feature_subgoal_idx = 0

        # === PLANNER V2: Start goal if enabled ===
        planner_v2_task_spec = None
        if planner_v2_enabled and planner_v2_adapter is not None:
            try:
                # Build context for planner
                planner_context = {
                    "repo_type": project_type.name if project_type else str(cfg.project_type),
                    "language": selected_buildpack_instance.buildpack_type.value
                        if selected_buildpack_instance else "unknown",
                    "test_cmd": effective_test_cmd,
                    "failing_test_file": v.failing_tests[0] if v.failing_tests else None,
                    "failing_tests": v.failing_tests[:5],
                    "error_signature": v.sig,
                }
                # Determine goal from mode
                if cfg.feature_mode:
                    goal = f"Implement feature: {cfg.description or 'new functionality'}"
                else:
                    goal = f"Fix failing tests: {len(v.failing_tests)} tests failing"
                
                planner_v2_task_spec = planner_v2_adapter.start_goal(goal, planner_context)
                log({
                    "phase": "planner_v2_goal_start",
                    "goal": goal,
                    "plan_id": planner_v2_adapter.get_plan().plan_id if planner_v2_adapter.get_plan() else None,
                    "first_step": planner_v2_task_spec.step_id,
                    "intent": planner_v2_task_spec.intent,
                })
            except Exception as e:
                log({"phase": "planner_v2_goal_start", "error": str(e)})

        # === PLANNER V5: Start planning if enabled ===
        planner_v5_initial_action = None
        if planner_v5_enabled and planner_v5_adapter is not None:
            try:
                # Build initial feedback for v5
                initial_feedback = {
                    "repo_url": cfg.github_url,
                    "failing_tests": v.failing_tests[:10],
                    "error_signature": v.sig,
                    "test_output": (v.stdout or "") + "\n" + (v.stderr or ""),
                    "tests_passed": 0,
                    "tests_failed": len(v.failing_tests),
                    "success": False,
                }
                
                # Get first action from planner v5
                planner_v5_initial_action = planner_v5_adapter.get_next_action(
                    controller_feedback=initial_feedback
                )
                
                if planner_v5_initial_action:
                    log({
                        "phase": "planner_v5_start",
                        "action_type": planner_v5_initial_action.action_type,
                        "mode": "v5_meta_planning",
                    })
            except Exception as e:
                log({"phase": "planner_v5_start", "error": str(e)})

        while step_count < max_iterations:
            # Progress reporting
            print(f"\n[Step {step_count}] Running tests...")
            v = _run_tests_in_sandbox(
                sb,
                effective_test_cmd,
                cfg,
                command_log,
                selected_buildpack,
                selected_buildpack_instance,
            )
            final_output = (v.stdout or "") + "\n" + (v.stderr or "")

            print(
                f"[Step {step_count}] Tests: {'PASS' if v.ok else 'FAIL'} | Failing: {len(v.failing_tests)} tests"
            )
            top_test_id = v.failing_tests[0] if v.failing_tests else None
            is_stalled = stall_state.update(len(v.failing_tests), top_test_id, v.sig)
            log(
                {
                    "phase": "measure",
                    "step": step_count,
                    "tests_ok": v.ok,
                    "exit_code": v.exit_code,
                    "failing_tests": v.failing_tests[:10],
                    "sig": v.sig,
                    "stalled": bool(is_stalled),
                }
            )

            # Initialize delta tracker on first test run (captures baseline)
            if delta_tracker is None and v.failing_tests:
                delta_tracker = TestDeltaTracker(set(v.failing_tests))
                log({
                    "phase": "delta_tracker_init",
                    "step": step_count,
                    "baseline_failing_count": len(v.failing_tests),
                })


            if v.ok:
                print(f"\n✅ SUCCESS! All tests passing after {step_count} steps.")
                current_phase = Phase.FINAL_VERIFY
                break

            # Track progress for early termination
            current_failing = len(v.failing_tests)
            if current_failing < min_failing_tests:
                min_failing_tests = current_failing
                steps_without_progress = 0
            else:
                steps_without_progress += 1

            if stall_state.iterations_without_improvement >= (stall_state.stall_threshold * 3):
                bailout_reason = (
                    f"Prolonged stall: {stall_state.iterations_without_improvement} iterations "
                    f"without improvement"
                )
                print(f"\n❌ Early termination: {bailout_reason}")
                log(
                    {
                        "phase": "bailout",
                        "step": step_count,
                        "reason": bailout_reason,
                        "sig": v.sig,
                        "top_test_id": top_test_id,
                    }
                )
                current_phase = Phase.BAILOUT
                break

            # Early termination: no progress after N steps
            if steps_without_progress >= cfg.max_steps_without_progress:
                print(f"\n❌ Early termination: No progress for {steps_without_progress} steps")
                bailout_reason = f"No progress for {steps_without_progress} steps"
                log(
                    {
                        "phase": "bailout",
                        "step": step_count,
                        "reason": bailout_reason,
                        "sig": v.sig,
                        "top_test_id": top_test_id,
                    }
                )
                current_phase = Phase.BAILOUT
                break

            # Track distinct signatures for multi-bug detection
            distinct_sigs.add(v.sig)
            if len(distinct_sigs) > 1:
                print(
                    f"[Step {step_count}] 🐛 Multi-bug detected: {len(distinct_sigs)} distinct error signatures"
                )

            # If stalled, force evidence gathering
            if is_stalled:
                log(
                    {
                        "phase": "stall_detected",
                        "step": step_count,
                        "sig": v.sig,
                        "iterations_without_improvement": stall_state.iterations_without_improvement,
                    }
                )
                print(f"[Step {step_count}] ⚠️  Stall detected - switching to evidence gathering")

            # controller policy
            pd = choose_policy(effective_test_cmd, v)
            if is_stalled:
                pd.intent = "gather_evidence"
                pd.subgoal = (
                    "Collect more context: list_tree, grep for error symbols, read new files"
                )

            if pd.confidence < 0.55 and steps_without_progress > 0:
                low_conf_streak += 1
            else:
                low_conf_streak = 0

            if low_conf_streak >= 4:
                bailout_reason = f"Confidence collapse: {low_conf_streak} low-confidence steps"
                print(f"\n❌ Early termination: {bailout_reason}")
                log(
                    {
                        "phase": "bailout",
                        "step": step_count,
                        "reason": bailout_reason,
                        "sig": v.sig,
                        "top_test_id": top_test_id,
                        "policy_confidence": pd.confidence,
                    }
                )
                current_phase = Phase.BAILOUT
                break

            if (
                tool_manager.total_requests_this_run >= cfg.max_tool_calls
                and pd.intent == "gather_evidence"
            ):
                bailout_reason = "Tool quota exhausted during evidence gathering"
                print(f"\n❌ Early termination: {bailout_reason}")
                log(
                    {
                        "phase": "bailout",
                        "step": step_count,
                        "reason": bailout_reason,
                        "tool_stats": tool_manager.get_stats(),
                    }
                )
                current_phase = Phase.BAILOUT
                break

            print(f"[Step {step_count}] Intent: {pd.intent} | Subgoal: {pd.subgoal[:60]}...")

            # gather high-signal files
            if is_quixbugs:
                files = _collect_relevant_files_quixbugs(sb, v, repo_tree_text)
            else:
                files = _collect_relevant_files(sb, v, repo_tree_text)
            files_block = _files_block(files)

            failing_test_file = normalize_test_path(v.failing_tests[0]) if v.failing_tests else None
            ctx = make_context_signature(
                failure_class=pd.intent,
                repo_type=project_type.name if project_type else str(cfg.project_type),
                language=selected_buildpack_instance.buildpack_type.value
                if selected_buildpack_instance
                else "unknown",
                env={
                    "docker_image": cfg.docker_image,
                    "unsafe_host_exec": bool(cfg.unsafe_host_exec),
                    "focus_timeout": int(cfg.focus_timeout),
                    "full_timeout": int(cfg.full_timeout),
                    "enable_sysdeps": bool(cfg.enable_sysdeps),
                },
                attempt_count=patch_attempts,
                failing_test_file=failing_test_file,
                sig=v.sig,
                stalled=bool(is_stalled),
            )
            action_priors_text = ""
            if memory_store is not None:
                priors = memory_store.query_action_priors(
                    ctx,
                    now_ts=int(clock.monotonic_steps()),
                )
                action_priors_text = format_action_priors(priors)
                log(
                    {
                        "phase": "learning_priors",
                        "step": step_count,
                        "context": ctx.as_dict(),
                        "priors": [
                            {
                                "action_type": p.action_type,
                                "action_key": p.action_key,
                                "weight": p.weight,
                                "success_rate": p.success_rate,
                                "mean_score": p.mean_score,
                                "n": p.n,
                            }
                            for p in priors
                        ],
                    }
                )

            # model state = facts
            if cfg.feature_mode:
                # Feature mode state - validate feature configuration
                if not cfg.feature_description:
                    print("[WARNING] Feature mode enabled but no feature description provided")

                if not cfg.acceptance_criteria:
                    print("[WARNING] Feature mode enabled but no acceptance criteria provided")

                # Optimize: compute subgoal once
                current_subgoal = (
                    feature_subgoals[current_feature_subgoal_idx]
                    if current_feature_subgoal_idx < len(feature_subgoals)
                    else "finalize: Review and complete feature"
                )

                # Build optional retrieval context if repo index is available
                retrieval_ctx = None
                if elite_repo_index is not None:
                    try:
                        retrieval_ctx = build_retrieval_context(
                            elite_repo_index,
                            (v.stdout or "") + "\n" + (v.stderr or ""),
                            max_files=8,
                            max_symbols=12,
                        )
                    except Exception:
                        retrieval_ctx = None

                state = {
                    "mode": MODE_FEATURE,
                    "goal": f"Implement feature: {cfg.feature_description or 'As specified'}",
                    "feature_description": cfg.feature_description or "",
                    "acceptance_criteria": cfg.acceptance_criteria or [],
                    "completed_subgoals": completed_feature_subgoals,  # Pass reference, not copy
                    "current_subgoal": current_subgoal,
                    "test_cmd": effective_test_cmd,
                    "focus_test_cmd": pd.focus_test_cmd,
                    "failure_output": (v.stdout or "") + "\n" + (v.stderr or ""),
                    "repo_tree": repo_tree_text,
                    "constraints": _constraints_text(),
                    "files_block": files_block,
                    "action_priors": action_priors_text,
                    "observations": observations,
                    "retrieval_context": retrieval_ctx,
                }
            else:
                # Repair mode state (original)
                # Build optional retrieval context if repo index is available
                retrieval_ctx = None
                if elite_repo_index is not None:
                    try:
                        retrieval_ctx = build_retrieval_context(
                            elite_repo_index,
                            (v.stdout or "") + "\n" + (v.stderr or ""),
                            max_files=8,
                            max_symbols=12,
                        )
                    except Exception:
                        retrieval_ctx = None

                state = {
                    "goal": "Make test command succeed (exit code 0).",
                    "intent": pd.intent,
                    "subgoal": pd.subgoal,
                    "test_cmd": effective_test_cmd,
                    "focus_test_cmd": pd.focus_test_cmd,
                    "failure_output": (v.stdout or "") + "\n" + (v.stderr or ""),
                    "repo_tree": repo_tree_text,
                    "constraints": _constraints_text(),
                    "files_block": files_block,
                    "retrieval_context": retrieval_ctx,
                    "action_priors": action_priors_text,
                    "observations": observations,
                }
            model_input = build_model_input(state)

            # ask model (try multiple temps for diversity)
            winner: Any = None
            patches_to_evaluate: list[tuple[str, float]] = []
            # Pre-fetch responses in parallel if enabled
            parallel_responses = []
            if cfg.parallel_patches:
                try:
                    import asyncio

                    from .llm.async_client import generate_patches_parallel
                    print(f"[Step {step_count}] Generating parallel responses (temps={cfg.temps})...")
                    parallel_responses = asyncio.run(
                        generate_patches_parallel(
                            model_input,
                            temperatures=cfg.temps,
                            model=cfg.model
                        )
                    )
                except Exception as e:
                    print(f"Parallel generation failed: {e}. Falling back to sync.")
                    parallel_responses = []

            for i, t in enumerate(cfg.temps):
                # Broadcast thinking state to dashboard
                broadcaster.thinking(True, f"Generating response (temp={t})")
                
                if i < len(parallel_responses):
                    resp = parallel_responses[i]
                    if resp.get("mode") == "error":
                         # Fallback to sync call on error
                         resp = call_ensemble_sync(
                             prompt=model_input,
                             temperature=t,
                             max_models=3 if cfg.ensemble_mode else 1,
                         )
                else:
                    resp = call_ensemble_sync(
                        prompt=model_input,
                        temperature=t,
                        max_models=3 if cfg.ensemble_mode else 1,
                    )
                
                broadcaster.thinking(False)
                log({"phase": "model", "step": step_count, "temp": t, "resp": resp})

                mode = resp.get("mode")
                if mode == "tool_request":
                    # execute requested tools; then continue to next iteration
                    tool_results = []
                    obs_additions = []
                    requests = resp.get("requests", [])[:6]

                    # Filter requests through tool manager
                    allowed_requests, blocked_reasons = tool_manager.filter_requests(requests)

                    for req in allowed_requests:
                        tool = req.get("tool", "")
                        args = req.get("args", {}) if isinstance(req.get("args"), dict) else {}
                        t0 = clock.perf_counter()
                        tr = _execute_tool(sb, tool, args)
                        clock.tick(1)
                        t1 = clock.perf_counter()
                        tool_results.append({"tool": tool, "args": args, "result": tr})
                        
                        # Broadcast tool execution to dashboard
                        tool_desc = f"{tool.replace('sandbox.', '')} - {'OK' if tr.get('ok') else 'FAIL'}"
                        broadcaster.tool(tool, tool_desc, args)

                        if memory_store is not None:
                            outcome = "success" if tr.get("ok") else "fail"
                            score = score_action(
                                outcome=outcome,
                                exec_time_ms=int((t1 - t0) * 1000.0),
                                command_count=1,
                                diff_lines=0,
                                regressions=0,
                            )
                            memory_store.record(
                                source_run_id=f"step{step_count}:temp{t}:tool{len(tool_results) - 1}",
                                context=ctx,
                                action_type="tool_request",
                                action_key=make_action_key_for_tool(tool, args),
                                action_json={"tool": tool, "args": args},
                                outcome=outcome,
                                score=score,
                                confidence_weight=1.0,
                                exec_time_ms=int((t1 - t0) * 1000.0),
                                command_count=1,
                                diff_lines=0,
                                regressions=0,
                            )

                        # Summarize for observations
                        summary = f"Tool: {tool}\n"
                        summary += f"Args: {args}\n"
                        summary += f"Exit: {tr.get('exit_code', 'N/A')}\n"
                        stdout = tr.get("stdout", "")[:500]
                        stderr = tr.get("stderr", "")[:500]
                        if stdout:
                            summary += f"Stdout: {stdout}\n"
                        if stderr:
                            summary += f"Stderr: {stderr}\n"
                        if tool == "sandbox.read_file" and tr.get("ok"):
                            content = tr.get("content", "")
                            summary += f"\n[File Content: {len(content)} bytes]\n"
                            summary += _truncate(content, 2000) + "\n"
                        if tool == "sandbox.grep" and tr.get("ok"):
                            matches = tr.get("matches", [])
                            if matches:
                                summary += f"Found {len(matches)} matches\n"
                        if tool == "sandbox.list_tree" and tr.get("ok"):
                            files = tr.get("files", [])
                            summary += f"Listed {len(files)} files\n"
                        obs_additions.append(summary)

                    log(
                        {
                            "phase": "tool_execution",
                            "step": step_count,
                            "temp": t,
                            "results": tool_results,
                            "blocked": blocked_reasons,
                        }
                    )

                    # Track tool call budget
                    total_tool_calls += len(allowed_requests)

                    # Provide structured feedback for blocked tool requests
                    if blocked_reasons:
                        for reason in blocked_reasons:
                            if reason:
                                # Build helpful feedback message
                                feedback = f"\n⚠️  BLOCKED TOOL REQUEST: {reason}\n"
                                if "shell idiom" in reason.lower() or "shell=False" in reason:
                                    feedback += "  → Split compound commands into separate tool_request entries\n"
                                    feedback += "  → Use simple commands only (no &&, ||, pipes, redirects)\n"
                                    feedback += (
                                        "  → Commands run from repo root - cd is not needed\n"
                                    )
                                elif (
                                    "allowlist" in reason.lower() or "not allowed" in reason.lower()
                                ):
                                    feedback += (
                                        "  → Use only allowed commands for this project type\n"
                                    )
                                    feedback += "  → Check project detection and available tools\n"
                                elif "quota" in reason.lower():
                                    feedback += f"  → Tool quota limit reached ({cfg.max_tool_calls} total)\n"
                                    feedback += "  → Focus on high-value operations only\n"
                                else:
                                    feedback += "  → Review command structure and arguments\n"

                                print(feedback)
                                observations += feedback

                    if not allowed_requests and any(
                        "Total tool request quota exceeded" in (br or "") for br in blocked_reasons
                    ):
                        # Prefer the tool manager's authoritative counter, fall back to our local tally.
                        used_calls = getattr(
                            tool_manager, "total_requests_this_run", total_tool_calls
                        )
                        bailout_reason = (
                            f"Tool quota exhausted ({used_calls}/{cfg.max_tool_calls} calls used)"
                        )
                        print(f"\n❌ Early termination: {bailout_reason}")
                        log(
                            {
                                "phase": "bailout",
                                "step": step_count,
                                "reason": bailout_reason,
                                "tool_stats": tool_manager.get_stats(),
                                "total_tool_calls": used_calls,
                            }
                        )
                        current_phase = Phase.BAILOUT
                        break

                    # Append to observations buffer
                    if obs_additions:
                        observations += "\n" + "\n".join(obs_additions)

                    # If we got tool requests, continue to next iteration
                    if allowed_requests:
                        break

                elif mode == "patch":
                    diff = resp.get("diff", "")
                    if diff:
                        dh = _diff_hash(diff)
                        if dh in bad_hashes:
                            print(f"[Step {step_count}] Skipping duplicate patch hash")
                            continue
                        bad_hashes.add(dh)
                        patches_to_evaluate.append((diff, t))

                elif mode == "feature_summary":
                    # Feature mode completion
                    summary = resp.get("summary", "")
                    completion_status = resp.get("completion_status", "")

                    print(f"\n[Step {step_count}] Feature summary received:")
                    print(f"Status: {completion_status}")
                    print(f"Summary: {summary[:200]}...")

                    log(
                        {
                            "phase": "feature_summary",
                            "step": step_count,
                            "completion_status": completion_status,
                            "summary": summary,
                        }
                    )

                    if completion_status == "complete":
                        # GATING: Do not accept "complete" without verification
                        # Run a quick verification to ensure tests pass
                        print(
                            f"\n[Step {step_count}] Feature claims completion - running verification..."
                        )
                        v_check = _run_tests_in_sandbox(
                            sb,
                            effective_test_cmd,
                            cfg,
                            command_log,
                            selected_buildpack,
                            selected_buildpack_instance,
                        )

                        if v_check.ok:
                            print(f"\n✅ FEATURE COMPLETE after {step_count} steps (verification passed)")
                            # Store summary for evidence pack and transition to FINAL_VERIFY
                            feature_summary = summary
                            current_phase = Phase.FINAL_VERIFY
                            break  # Exit repair loop to run FINAL_VERIFY
                        else:
                            # Verification failed - force back into repair loop
                            feedback = (
                                f"\n⚠️  COMPLETION REJECTED: Verification failed with {len(v_check.failing_tests)} "
                                "failing tests.\n"
                                "  → Cannot mark completion_status='complete' until all tests pass\n"
                                "  → Continue implementing and fixing until verification succeeds\n"
                                f"  → Failing tests: {v_check.failing_tests[:5]}\n"
                            )
                            print(feedback)
                            observations += feedback
                            log(
                                {
                                    "phase": "feature_completion_rejected",
                                    "step": step_count,
                                    "reason": "verification_failed",
                                    "failing_tests": v_check.failing_tests[:10],
                                }
                            )
                            # Continue loop - do not accept premature completion
                    elif completion_status == "blocked":
                        bailout_reason = f"Feature blocked: {summary[:100]}"
                        print(f"\n❌ Early termination: {bailout_reason}")
                        log(
                            {
                                "phase": "bailout",
                                "step": step_count,
                                "reason": bailout_reason,
                            }
                        )
                        current_phase = Phase.BAILOUT
                        break
                    # For "partial" or "in_progress", continue iteration

            if current_phase == Phase.BAILOUT:
                break

            # Evaluate patches
            if patches_to_evaluate:
                patch_attempts += 1
                print(f"[Step {step_count}] Evaluating {len(patches_to_evaluate)} patch(es)...")

                # Validate patch hygiene
                valid_patches: list[tuple[str, float]] = []

                # Choose hygiene config based on mode and language
                # Extract language from detection result
                detected_language = None
                if selected_buildpack_instance:
                    detected_language = selected_buildpack_instance.buildpack_type.value
                elif project_type:
                    detected_language = project_type.name.lower()

                # Get adaptive limits from budget controller
                budget_max_lines, budget_max_files = patch_budget.get_limits()
                
                # Use budget controller limits, with feature mode adjustments
                if cfg.feature_mode:
                    # Feature mode may need higher baseline, but still use budget controller
                    base_config = PatchHygieneConfig.for_feature_mode(language=detected_language)
                    # Use the max of budget controller and feature mode defaults
                    hygiene_config = PatchHygieneConfig.custom(
                        max_lines_changed=max(budget_max_lines, base_config.max_lines_changed),
                        max_files_changed=max(budget_max_files, base_config.max_files_changed),
                        allow_test_modification=True,
                        language=detected_language,
                    )
                else:
                    # Repair mode uses budget controller limits directly
                    hygiene_config = PatchHygieneConfig.custom(
                        max_lines_changed=budget_max_lines,
                        max_files_changed=budget_max_files,
                        language=detected_language,
                    )

                # Apply CLI overrides
                if cfg.max_lines_changed is not None:
                    hygiene_config.max_lines_changed = cfg.max_lines_changed
                if cfg.max_files_changed is not None:
                    hygiene_config.max_files_changed = cfg.max_files_changed
                if cfg.allow_lockfile_changes:
                    hygiene_config.allow_lockfile_changes = True

                log(
                    {
                        "phase": "hygiene_policy",
                        "step": step_count,
                        "mode": "feature" if cfg.feature_mode else "repair",
                        "language": detected_language,
                        "max_lines": hygiene_config.max_lines_changed,
                        "max_files": hygiene_config.max_files_changed,
                        "allow_lockfile_changes": hygiene_config.allow_lockfile_changes,
                        "budget_tier": patch_budget.current_tier.name,
                    }
                )

                for diff, temp in patches_to_evaluate:
                    # Minimize diff before hygiene check
                    minimized = diff_minimizer.minimize(diff)
                    if minimized.dropped_hunks > 0:
                        log({
                            "phase": "diff_minimized",
                            "step": step_count,
                            "dropped_hunks": minimized.dropped_hunks,
                            "reduction_ratio": round(minimized.reduction_ratio, 2),
                        })
                        diff_to_check = minimized.minimized
                    else:
                        diff_to_check = diff
                    
                    hygiene_result = validate_patch_hygiene(diff_to_check, hygiene_config)
                    if hygiene_result.is_valid:
                        valid_patches.append((diff, temp))
                    else:
                        print(
                            f"[Step {step_count}] Patch rejected by hygiene gates: {hygiene_result.violations}"
                        )
                        if memory_store is not None:
                            action_json = make_action_json_for_patch(diff)
                            memory_store.record(
                                source_run_id=f"step{step_count}:temp{temp}:patch_hygiene_reject",
                                context=ctx,
                                action_type="patch",
                                action_key=make_action_key_for_patch(diff),
                                action_json=action_json,
                                outcome="blocked",
                                score=0.0,
                                confidence_weight=1.0 / (1.0 + float(temp)),
                                exec_time_ms=0,
                                command_count=0,
                                diff_lines=int(action_json.get("diff_lines", 0)),
                                regressions=0,
                            )
                        log(
                            {
                                "phase": "patch_rejected",
                                "step": step_count,
                                "reasons": hygiene_result.violations,
                            }
                        )

                if valid_patches:
                    # Track patch attempts budget
                    total_patch_attempts += len(valid_patches)

                    # Check patch attempt budget
                    if total_patch_attempts > cfg.max_patch_attempts:
                        bailout_reason = (
                            f"Patch attempt budget exhausted ({total_patch_attempts}/{cfg.max_patch_attempts})"
                        )
                        print(f"\n❌ Early termination: {bailout_reason}")
                        log(
                            {
                                "phase": "bailout",
                                "step": step_count,
                                "reason": bailout_reason,
                                "total_patch_attempts": total_patch_attempts,
                            }
                        )
                        current_phase = Phase.BAILOUT
                        break

                    # Evaluate in parallel worktrees
                    results = evaluate_patches_parallel(
                        sb,
                        valid_patches,
                        pd.focus_test_cmd,
                        effective_test_cmd,
                        docker_image=cfg.docker_image,
                        cpu=cfg.cpu,
                        mem_mb=cfg.mem_mb,
                        durability_reruns=cfg.durability_reruns,
                        unsafe_host_exec=cfg.unsafe_host_exec,
                    )

                    if memory_store is not None:
                        for patch_res in results:
                            action_json = make_action_json_for_patch(patch_res.diff)
                            outcome = "success" if patch_res.ok else "fail"
                            score = score_action(
                                outcome=outcome,
                                exec_time_ms=0,
                                command_count=2,
                                diff_lines=int(action_json.get("diff_lines", 0)),
                                regressions=0,
                            )
                            memory_store.record(
                                source_run_id=f"step{step_count}:temp{patch_res.temperature}:patch_eval:{patch_res.diff_hash}",
                                context=ctx,
                                action_type="patch",
                                action_key=patch_res.diff_hash,
                                action_json=action_json,
                                outcome=outcome,
                                score=score,
                                confidence_weight=1.0 / (1.0 + float(patch_res.temperature)),
                                exec_time_ms=0,
                                command_count=2,
                                diff_lines=int(action_json.get("diff_lines", 0)),
                                regressions=0,
                            )

                    winner = find_first_successful_patch(results)
                    if winner:
                        print(f"[Step {step_count}] ✅ Found winning patch!")
                        log(
                            {
                                "phase": "winner_found",
                                "step": step_count,
                                "winner_hash": getattr(winner, "diff_hash", ""),
                            }
                        )
                        
                        # QA gate check before applying winner
                        # NOTE: If winner came from evaluate_patches_parallel, tests already passed
                        # in the worktree verification. We trust that result and apply the patch.
                        # The QA gate's rule-based fallback would reject asking for "test evidence"
                        # that we already have from parallel evaluation.
                        qa_passed = True
                        
                        # Only run QA gate if we have an LLM-based QA (not rule-based fallback)
                        # since rule-based always challenges for test evidence we already have
                        if qa_orchestrator is not None and qa_orchestrator.has_llm_critic():
                            try:
                                # Update QA config with current patch budget limits
                                current_limits = patch_budget.get_limits()
                                qa_orchestrator.config.surgical_max_lines = current_limits[0]
                                qa_orchestrator.config.surgical_max_files = current_limits[1]
                                # Update delta tracker reference
                                qa_orchestrator.collector.delta_tracker = delta_tracker
                                
                                qa_result = qa_orchestrator.evaluate_patch(
                                    diff=getattr(winner, "diff", ""),
                                    failing_tests=pd.failing_tests if hasattr(pd, "failing_tests") else [],
                                    test_cmd=effective_test_cmd,
                                    failure_signature=getattr(ctx, "error_signature", "") if ctx else "",
                                )
                                log({
                                    "phase": "qa_gate",
                                    "step": step_count,
                                    "accepted": qa_result.accepted,
                                    "rejection_reasons": qa_result.rejection_reasons,
                                    "escalation_tags": qa_result.escalation_tags,
                                })
                                
                                if not qa_result.accepted:
                                    qa_passed = False
                                    print(f"[Step {step_count}] ⚠️ QA gate rejected: {qa_result.rejection_reasons}")
                                    
                                    # Check if we should escalate patch budget
                                    if qa_orchestrator.should_escalate_budget(qa_result):
                                        # Log before escalation for context
                                        log({
                                            "phase": "patch_budget_qa_escalation_trigger",
                                            "step": step_count,
                                            "stagnation_count": steps_without_progress,
                                        })
                                        escalated = patch_budget.escalate()
                                        if escalated:
                                            log({
                                                "phase": "patch_budget_qa_escalate",
                                                "step": step_count,
                                                **patch_budget.get_state_summary(),
                                            })
                                elif qa_result.escalation_tags:
                                    tags = qa_result.escalation_tags
                                    print(f"[Step {step_count}] ⚠️ QA accepted with escalations: {tags}")
                            except Exception as qa_err:
                                log({"phase": "qa_gate_error", "step": step_count, "error": str(qa_err)})
                                # Continue anyway if QA fails
                        
                        if qa_passed:
                            # Apply winner to main repo
                            apply_patch(sb, getattr(winner, "diff", ""))
                            winner_diff = getattr(winner, "diff", "")

                        # In feature mode, progress through subgoals ONLY if patch is successful
                        # The winner patch passed verification, so we can mark current subgoal as complete
                        if cfg.feature_mode and current_feature_subgoal_idx < len(feature_subgoals):
                            completed_subgoal = feature_subgoals[current_feature_subgoal_idx]
                            completed_feature_subgoals.append(completed_subgoal)
                            current_feature_subgoal_idx += 1
                            print(f"[Step {step_count}] ✓ Completed subgoal: {completed_subgoal}")
                            log(
                                {
                                    "phase": "feature_subgoal_complete",
                                    "step": step_count,
                                    "completed_subgoal": completed_subgoal,
                                    "remaining_subgoals": len(feature_subgoals)
                                    - current_feature_subgoal_idx,
                                }
                            )

                            # If all subgoals completed, check for feature completion
                            if current_feature_subgoal_idx >= len(feature_subgoals):
                                print(
                                    f"[Step {step_count}] All subgoals completed - awaiting feature summary"
                                )
                                # Continue to next iteration to get feature_summary from model
                            # Don't break - continue to next subgoal
                        else:
                            # Repair mode or feature mode complete - break to final verify
                            current_phase = Phase.FINAL_VERIFY
                            break
                    else:
                        print(f"[Step {step_count}] No patch passed verification")
                        for res in results:
                            if not res.ok:
                                print(f"  > Patch {res.diff_hash[:8]} failed: {res.info[:200]}...")
                        log({"phase": "no_winner", "step": step_count, "attempted": len(valid_patches)})
                        
                        # Update patch budget controller with failure
                        patch_budget.record_attempt(
                            failing_tests=set(v.failing_tests),
                            success=False,
                        )
                        
                        # Check if we should escalate limits
                        if patch_budget.should_escalate():
                            if patch_budget.escalate():
                                log({
                                    "phase": "budget_escalated",
                                    "step": step_count,
                                    **patch_budget.get_state_summary(),
                                })
                                print(f"[Step {step_count}] ⬆️  Escalated to {patch_budget.current_tier.name} "
                                      f"(lines: {patch_budget.get_limits()[0]})")
                        
                        # Track with termination heuristics
                        if termination_heuristics is not None:
                            for diff, _ in valid_patches:
                                termination_heuristics.record_attempt(diff, success=False)
                            
                            # Check if we should terminate early
                            should_terminate, term_reason = termination_heuristics.should_terminate()
                            if should_terminate:
                                bailout_reason = f"Early termination: {term_reason}"
                                print(f"\n❌ {bailout_reason}")
                                log({
                                    "phase": "early_termination",
                                    "step": step_count,
                                    "reason": term_reason,
                                })
                                current_phase = Phase.BAILOUT
                                break


            step_count += 1
            
            # Log controller step event
            try:
                from .events import log_controller_step_global
                log_controller_step_global(
                    step_number=step_count,
                    phase=current_phase.name if hasattr(current_phase, 'name') else str(current_phase),
                    data={
                        "tests_ok": v.ok,
                        "failing_tests_count": len(v.failing_tests),
                        "steps_without_progress": steps_without_progress,
                    }
                )
            except ImportError:
                pass  # Events module not available

            # === PLANNER V2: Process step outcome and get next step ===
            if planner_v2_enabled and planner_v2_adapter is not None and planner_v2_task_spec is not None:
                try:
                    from .planner_v2 import ControllerOutcome
                    # Build outcome from this step
                    outcome = ControllerOutcome(
                        step_id=planner_v2_task_spec.step_id,
                        success=v.ok or (len(v.failing_tests) < min_failing_tests),
                        patch_applied=len(valid_patches) > 0 if 'valid_patches' in dir() else False,
                        tests_passed=v.ok,
                        error_message=v.sig if not v.ok else None,
                        metrics={
                            "step_count": step_count,
                            "failing_tests": len(v.failing_tests),
                            "exit_code": v.exit_code,
                        }
                    )
                    
                    # Process outcome and get next step
                    planner_v2_task_spec = planner_v2_adapter.process_outcome(outcome)
                    
                    if planner_v2_task_spec is not None:
                        log({
                            "phase": "planner_v2_next_step",
                            "step": step_count,
                            "next_step_id": planner_v2_task_spec.step_id,
                            "intent": planner_v2_task_spec.intent,
                        })
                    else:
                        # Plan complete or halted
                        summary = planner_v2_adapter.get_summary()
                        log({
                            "phase": "planner_v2_complete",
                            "step": step_count,
                            "summary": summary,
                        })
                        if planner_v2_adapter.is_halted():
                            bailout_reason = f"Planner v2 halted: {planner_v2_adapter.get_halt_reason()}"
                            log({"phase": "bailout", "reason": bailout_reason})
                            current_phase = Phase.BAILOUT
                            break
                except Exception as e:
                    log({"phase": "planner_v2_process_outcome", "error": str(e)})

            # === PLANNER V5: Process outcome and get next action ===
            if planner_v5_enabled and planner_v5_adapter is not None:
                try:
                    # Build feedback from this step
                    v5_feedback = {
                        "success": v.ok,
                        "output": (v.stdout or "") + "\n" + (v.stderr or ""),
                        "tests_passed": len([t for t in v.failing_tests]) if not v.ok else 0,
                        "tests_failed": len(v.failing_tests),
                        "traceback": v.stderr if not v.ok else None,
                    }
                    
                    # Get next action from planner v5
                    next_action = planner_v5_adapter.get_next_action(
                        controller_feedback=v5_feedback
                    )
                    
                    if next_action:
                        log({
                            "phase": "planner_v5_next_action",
                            "step": step_count,
                            "action_type": next_action.action_type,
                            "target": next_action.target_path or next_action.command,
                        })
                        # Store for potential use in next iteration
                        planner_v5_initial_action = next_action
                    else:
                        log({
                            "phase": "planner_v5_no_action",
                            "step": step_count,
                        })
                except Exception as e:
                    log({"phase": "planner_v5_process_outcome", "error": str(e)})

        # === PHASE: FINAL_VERIFY ===
        if current_phase == Phase.FINAL_VERIFY:
            log(PhaseTransition(Phase.REPAIR_LOOP, Phase.FINAL_VERIFY).to_dict())

            # Track verification results
            verification_passed = True
            verification_results = []
            
            broadcaster.status("FINAL_VERIFY", step=step_count)

            # Execute verification commands based on policy
            run_cmd_verification = cfg.verify_policy in ("cmds_then_tests", "cmds_only")

            # 1. Focused verify commands (if any)
            if run_cmd_verification:
                for idx, cmd in enumerate(cfg.focused_verify_cmds):
                    total_verification_attempts += 1

                    # Check verification budget
                    if total_verification_attempts > cfg.max_verification_attempts:
                        bailout_reason = (
                            f"Verification attempt budget exhausted "
                            f"({total_verification_attempts}/{cfg.max_verification_attempts})"
                        )
                        print(f"\n❌ Early termination: {bailout_reason}")
                        log(
                            {
                                "phase": "bailout",
                                "reason": bailout_reason,
                                "total_verification_attempts": total_verification_attempts,
                            }
                        )
                        verification_passed = False
                        break

                    print(
                        f"\n[FINAL_VERIFY] Running focused verification {idx + 1}/{len(cfg.focused_verify_cmds)}: {cmd}"
                    )
                    v_result = _run_tests_in_sandbox(
                        sb, cmd, cfg, command_log, selected_buildpack, selected_buildpack_instance
                    )
                    verification_results.append(
                        {
                            "type": "focused_verify",
                            "command": cmd,
                            "passed": v_result.ok,
                            "exit_code": v_result.exit_code,
                        }
                    )
                    log(
                        {
                            "phase": "focused_verify",
                            "command": cmd,
                            "passed": v_result.ok,
                            "exit_code": v_result.exit_code,
                        }
                    )
                    if not v_result.ok:
                        print("  ❌ Focused verification failed")
                        verification_passed = False
                    else:
                        print("  ✅ Focused verification passed")

            # 2. Regular verify commands (if any)
            if run_cmd_verification:
                for idx, cmd in enumerate(cfg.verify_cmds):
                    total_verification_attempts += 1

                    # Check verification budget
                    if total_verification_attempts > cfg.max_verification_attempts:
                        bailout_reason = (
                            f"Verification attempt budget exhausted "
                            f"({total_verification_attempts}/{cfg.max_verification_attempts})"
                        )
                        print(f"\n❌ Early termination: {bailout_reason}")
                        log(
                            {
                                "phase": "bailout",
                                "reason": bailout_reason,
                                "total_verification_attempts": total_verification_attempts,
                            }
                        )
                        verification_passed = False
                        break

                    print(
                        f"\n[FINAL_VERIFY] Running verification {idx + 1}/{len(cfg.verify_cmds)}: {cmd}"
                    )
                    v_result = _run_tests_in_sandbox(
                        sb, cmd, cfg, command_log, selected_buildpack, selected_buildpack_instance
                    )
                    verification_results.append(
                        {
                            "type": "verify",
                            "command": cmd,
                            "passed": v_result.ok,
                            "exit_code": v_result.exit_code,
                        }
                    )
                    log(
                        {
                            "phase": "verify",
                            "command": cmd,
                            "passed": v_result.ok,
                            "exit_code": v_result.exit_code,
                        }
                    )
                    if not v_result.ok:
                        print("  ❌ Verification failed")
                        verification_passed = False
                    else:
                        print("  ✅ Verification passed")

            # 3. Test command (unless policy is "cmds_only")
            if cfg.verify_policy != "cmds_only":
                # Run tests with reproducibility check (N times if configured)
                # Ensure at least 1 run
                repro_times = max(1, cfg.repro_times)
                repro_passed = True
                for run_idx in range(repro_times):
                    # Track verification attempts
                    total_verification_attempts += 1

                    # Check verification budget
                    if total_verification_attempts > cfg.max_verification_attempts:
                        bailout_reason = (
                            f"Verification attempt budget exhausted "
                            f"({total_verification_attempts}/{cfg.max_verification_attempts})"
                        )
                        print(f"\n❌ Early termination: {bailout_reason}")
                        log(
                            {
                                "phase": "bailout",
                                "reason": bailout_reason,
                                "total_verification_attempts": total_verification_attempts,
                            }
                        )
                        verification_passed = False
                        # Set v and final_output to reflect budget exhaustion
                        v = VerifyResult(
                            ok=False,
                            exit_code=1,
                            stdout="",
                            stderr="Budget exhausted",
                            failing_tests=[],
                            sig="",
                        )
                        final_output = "Budget exhausted"
                        break

                    if repro_times > 1:
                        print(
                            f"\n[FINAL_VERIFY] Running test suite (run {run_idx + 1}/{repro_times}): "
                            f"{effective_test_cmd}"
                        )
                    else:
                        print(f"\n[FINAL_VERIFY] Running test suite: {effective_test_cmd}")

                    v = _run_tests_in_sandbox(
                        sb,
                        effective_test_cmd,
                        cfg,
                        command_log,
                        selected_buildpack,
                        selected_buildpack_instance,
                    )

                    # Always store final output from current run (in case we break early)
                    final_output = (v.stdout or "") + "\n" + (v.stderr or "")

                    verification_results.append(
                        {
                            "type": "tests",
                            "command": effective_test_cmd,
                            "passed": v.ok,
                            "exit_code": v.exit_code,
                            "run": run_idx + 1,
                        }
                    )
                    log(
                        {
                            "phase": "test_verify",
                            "command": effective_test_cmd,
                            "passed": v.ok,
                            "exit_code": v.exit_code,
                            "run": run_idx + 1,
                        }
                    )

                    if not v.ok:
                        print(
                            f"  ❌ Tests failed (run {run_idx + 1}): {len(v.failing_tests)} failing tests"
                        )
                        verification_passed = False
                        repro_passed = False
                        if repro_times > 1:
                            print("  ⚠️  Stopping reproducibility check due to failure")
                            break
                    else:
                        print(f"  ✅ Tests passed (run {run_idx + 1})")

                if repro_passed and cfg.repro_times > 1:
                    print(
                        f"\n✅ Test suite is reproducible ({cfg.repro_times}/{cfg.repro_times} runs passed)"
                    )
            else:
                # For cmds_only policy, create a placeholder result
                v = VerifyResult(
                    ok=verification_passed,
                    exit_code=0 if verification_passed else 1,
                    stdout="",
                    stderr="",
                    failing_tests=[],
                    sig="",
                )
                final_output = ""

            # Overall verification result
            log(
                {
                    "phase": "final_verify_complete",
                    "verification_passed": verification_passed,
                    "verification_results": verification_results,
                }
            )

            if verification_passed:
                print("\n✅ FINAL SUCCESS! All verifications passed.")
                broadcaster.log("Run completed successfully", level="success")
                broadcaster.status("SUCCESS")
            else:
                print("\n⚠️  Final verify failed")
                broadcaster.log("Run failed verification", level="error")
                broadcaster.status("FAILURE")
                current_phase = Phase.BAILOUT

        # === PHASE: EVIDENCE_PACK ===
        current_phase = Phase.EVIDENCE_PACK
        log(PhaseTransition(Phase.FINAL_VERIFY, Phase.EVIDENCE_PACK).to_dict())

        # Export evidence pack
        state_dict = {
            "config": cfg.__dict__,
            "project_type": project_type.name if project_type else None,
            "setup_commands": setup_commands,
            "effective_test_cmd": effective_test_cmd,
            "steps_taken": step_count,
            "patch_attempts": patch_attempts,
            "min_failing_tests": min_failing_tests,
            "final_failing_tests": len(v.failing_tests) if not v.ok else 0,
            "final_ok": v.ok,
            "bailout_reason": bailout_reason,
            "feature_summary": feature_summary,  # Include feature summary if present
            # Budget tracking
            "total_tool_calls": total_tool_calls,
            "total_patch_attempts": total_patch_attempts,
            "total_verification_attempts": total_verification_attempts,
            "budgets": {
                "max_tool_calls": cfg.max_tool_calls,
                "max_patch_attempts": cfg.max_patch_attempts,
                "max_verification_attempts": cfg.max_verification_attempts,
            },
        }

        pack_dir = evidence_exporter.export(
            sandbox_root=str(sb.root) if sb.root else "",
            log_dir=log_dir,
            baseline_output=baseline_output,
            final_output=final_output,
            winner_diff=winner_diff,
            state=state_dict,
            command_log=command_log,
            run_id=run_id,
        )

        print(f"\n[EVIDENCE_PACK] Exported to: {pack_dir}")

        return {
            "ok": v.ok,
            "sandbox": sb.root,
            "repo_dir": sb.repo_dir,
            "steps_taken": step_count,
            "evidence_pack": pack_dir,
            "winner_diff": winner_diff,
        }

    except Exception as e:
        # Fail-closed: Create evidence pack even on exception
        import traceback

        error_details = traceback.format_exc()

        # Log error if log directory exists
        if log_dir:
            try:
                write_jsonl(
                    log_dir,
                    {
                        "phase": "exception",
                        "error": str(e),
                        "traceback": error_details,
                    },
                    clock=clock,
                )
            except Exception:
                pass  # If logging fails, continue with error reporting

        # Try to create a minimal evidence pack
        evidence_pack_path = None
        try:
            # Create minimal state dict with available information
            state_dict = {
                "config": cfg.__dict__ if cfg else {},
                "project_type": None,
                "setup_commands": [],
                "effective_test_cmd": None,
                "steps_taken": 0,
                "error": str(e),
                "traceback": error_details,
                "bailout_reason": f"Exception: {type(e).__name__}: {e!s}",
            }

            evidence_pack_path = evidence_exporter.export(
                sandbox_root=sb.root if sb else "",
                log_dir=str(log_dir) if log_dir else "",
                baseline_output="",
                final_output="",
                winner_diff=None,
                state=state_dict,
                command_log=command_log,
                run_id=run_id,
            )
            print(f"\n[EXCEPTION] Evidence pack created at: {evidence_pack_path}")
        except Exception as pack_error:
            print(f"\n[EXCEPTION] Failed to create evidence pack: {pack_error}")

        return {
            "ok": False,
            "error": f"Exception: {type(e).__name__}: {e!s}",
            "traceback": error_details,
            "sandbox": sb.root if sb else None,
            "repo_dir": sb.repo_dir if sb else None,
            "evidence_pack": evidence_pack_path,
        }

    finally:
        if 'broadcaster' in locals():
            broadcaster.close()
        if memory_store is not None:
            memory_store.close()


def _run_tests_in_sandbox(
    sb: Sandbox,
    test_cmd: str,
    cfg: ControllerConfig,
    command_log: list[dict[str, Any]],
    docker_image: str,
    buildpack_instance=None,
) -> VerifyResult:
    """Run tests in Docker or on host based on configuration.

    Args:
        sb: The sandbox.
        test_cmd: Test command to run.
        cfg: Controller configuration.
        command_log: Command execution log.
        docker_image: Docker image to use for execution.
        buildpack_instance: Optional buildpack instance for failure parsing.

    Returns:
        VerifyResult with test results.
    """
    if cfg.unsafe_host_exec:
        # Run on host (unsafe)
        return run_tests(sb, test_cmd, timeout_sec=cfg.focus_timeout)
    else:
        # Run in Docker with network OFF
        result = docker_test(sb, test_cmd, timeout_sec=cfg.focus_timeout, docker_image=docker_image)

        command_log.append(
            {
                "phase": "test",
                "command": test_cmd,
                "exit_code": result.exit_code,
                "ok": result.ok,
                "stdout": result.stdout[:2000],
                "stderr": result.stderr[:2000],
                "timed_out": result.timed_out,
            }
        )

        # Use buildpack's parse_failures if available, otherwise fallback to legacy parser
        if buildpack_instance and hasattr(buildpack_instance, "parse_failures"):
            failure_info = buildpack_instance.parse_failures(result.stdout, result.stderr)
            failing_tests = failure_info.failing_tests
            sig = failure_info.signature
        else:
            from .parsers import error_signature, parse_pytest_failures

            failing_tests = parse_pytest_failures(result.stdout + result.stderr)
            sig = error_signature(result.stdout, result.stderr)

        return VerifyResult(
            ok=result.ok,
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
            failing_tests=failing_tests,
            sig=sig,
        )
