"""CLI entry point for CGW-based coding agent.
from __future__ import annotations

This module provides a command-line interface to run the serial decision
coding agent through the CGW/SSL guard architecture.

Usage:
    python -m rfsn_controller.cgw_cli --repo https://github.com/user/repo --test "pytest -q"
    
    # Or with CGW mode flag if using the main CLI:
    python -m rfsn_controller.cli --repo ... --cgw-mode
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from .cgw_bridge import BridgeConfig, CGWControllerBridge

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for CGW mode."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def run_cgw_agent(
    github_url: str,
    test_cmd: str = "pytest -q",
    max_cycles: int = 50,
    max_patches: int = 10,
    log_events: bool = True,
    event_log_path: Path | None = None,
    verbose: bool = False,
) -> dict:
    """Run the CGW coding agent.
    
    Args:
        github_url: Public GitHub repository URL.
        test_cmd: Test command to run.
        max_cycles: Maximum decision cycles.
        max_patches: Maximum patches to apply.
        log_events: Whether to log events for replay.
        event_log_path: Path to save event log JSON.
        verbose: Enable verbose logging.
    
    Returns:
        Dictionary with results and event log.
    """
    setup_logging(verbose)
    
    logger.info(f"Starting CGW coding agent for {github_url}")
    logger.info(f"Test command: {test_cmd}")
    logger.info(f"Max cycles: {max_cycles}, Max patches: {max_patches}")
    
    # Create bridge config
    config = BridgeConfig(
        github_url=github_url,
        test_cmd=test_cmd,
        max_cycles=max_cycles,
        max_patches=max_patches,
        log_events=log_events,
        event_log_path=event_log_path,
    )
    
    # Create sandbox (simplified - in production would clone repo)
    # For now we run without a real sandbox for demonstration
    sandbox = None
    
    # Create and run bridge
    bridge = CGWControllerBridge(config=config, sandbox=sandbox)
    result = bridge.run()
    
    # Save event log if path specified
    if event_log_path and log_events:
        event_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(event_log_path, "w") as f:
            json.dump(result["event_log"], f, indent=2, default=str)
        logger.info(f"Event log saved to {event_log_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("CGW CODING AGENT RESULT")
    print("=" * 60)
    print(f"Success: {result['success']}")
    print(f"Final action: {result['final_action']}")
    print(f"Cycles executed: {result['cycles_executed']}")
    print(f"Total time: {result['total_time_ms']:.1f}ms")
    print(f"Tests passing: {result['tests_passing']}")
    print(f"Patches applied: {result['patches_applied']}")
    print(f"Seriality maintained: {result['seriality_maintained']}")
    if result.get("error"):
        print(f"Error: {result['error']}")
    print("=" * 60)
    
    return result


def main() -> None:
    """CLI entry point for CGW coding agent."""
    load_dotenv()
    
    parser = argparse.ArgumentParser(
        description="Serial Decision Coding Agent using CGW/SSL Guard Architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python -m rfsn_controller.cgw_cli --repo https://github.com/user/repo

    # With custom test command and limits
    python -m rfsn_controller.cgw_cli \\
        --repo https://github.com/user/repo \\
        --test "pytest tests/ -q" \\
        --max-cycles 100 \\
        --max-patches 20

    # With event logging for replay
    python -m rfsn_controller.cgw_cli \\
        --repo https://github.com/user/repo \\
        --save-events ./events.json
        """,
    )
    
    # Required arguments
    parser.add_argument(
        "--repo",
        required=True,
        help="Public GitHub URL: https://github.com/OWNER/REPO",
    )
    
    # Test configuration
    parser.add_argument(
        "--test",
        default="pytest -q",
        help="Test command to satisfy (default: pytest -q)",
    )
    
    # CGW-specific settings
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=50,
        help="Maximum decision cycles (default: 50)",
    )
    parser.add_argument(
        "--max-patches",
        type=int,
        default=10,
        help="Maximum patches to apply (default: 10)",
    )
    
    # Event logging
    parser.add_argument(
        "--save-events",
        type=Path,
        default=None,
        help="Path to save event log JSON for replay",
    )
    parser.add_argument(
        "--no-events",
        action="store_true",
        help="Disable event logging",
    )
    
    # Verbosity
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    # Output format
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON",
    )
    
    args = parser.parse_args()
    
    try:
        result = run_cgw_agent(
            github_url=args.repo,
            test_cmd=args.test,
            max_cycles=args.max_cycles,
            max_patches=args.max_patches,
            log_events=not args.no_events,
            event_log_path=args.save_events,
            verbose=args.verbose,
        )
        
        if args.json:
            # Remove large fields for JSON output
            output = {k: v for k, v in result.items() if k != "event_log"}
            print(json.dumps(output, indent=2, default=str))
        
        # Exit with appropriate code
        sys.exit(0 if result["success"] else 1)
        
    except Exception as e:
        logger.exception(f"CGW agent failed: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
