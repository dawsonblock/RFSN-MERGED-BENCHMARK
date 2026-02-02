#!/usr/bin/env python3
"""CGW Coding Agent CLI.

Command-line interface for the CGW Serial Decision Coding Agent.

Usage examples:
    # Run with default config
    python -m cgw_ssl_guard.coding_agent.cli --goal "Fix failing tests"
    
    # Run with config file
    python -m cgw_ssl_guard.coding_agent.cli --config cgw.yaml
    
    # Run with repository URL
    python -m cgw_ssl_guard.coding_agent.cli --repo https://github.com/user/repo.git
    
    # Run with dashboard
    python -m cgw_ssl_guard.coding_agent.cli --goal "Fix tests" --dashboard
    
    # Generate default config
    python -m cgw_ssl_guard.coding_agent.cli --init-config
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

# Configure logging early
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="cgw-agent",
        description="CGW Serial Decision Coding Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --goal "Fix failing tests"
  %(prog)s --config cgw.yaml --repo https://github.com/user/repo.git
  %(prog)s --dashboard --max-cycles 50
  %(prog)s --init-config > cgw.yaml
        """,
    )
    
    # Core options
    parser.add_argument(
        "--goal", "-g",
        type=str,
        default="Fix failing tests",
        help="Goal description for the agent (default: 'Fix failing tests')",
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to YAML/JSON config file",
    )
    
    parser.add_argument(
        "--repo", "-r",
        type=str,
        default=None,
        help="Git repository URL to clone",
    )
    
    parser.add_argument(
        "--branch", "-b",
        type=str,
        default="main",
        help="Git branch to use (default: main)",
    )
    
    # Limits
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="Maximum decision cycles (default: from config)",
    )
    
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Total timeout in seconds (default: from config)",
    )
    
    # Dashboard
    parser.add_argument(
        "--dashboard", "-d",
        action="store_true",
        help="Enable web dashboard",
    )
    
    parser.add_argument(
        "--dashboard-port",
        type=int,
        default=8765,
        help="Dashboard HTTP port (default: 8765)",
    )
    
    # Output
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Session ID for event tracking (auto-generated if not provided)",
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file for session events (JSON)",
    )
    
    # Misc
    parser.add_argument(
        "--verbose", "-v",
        action="count",
        default=0,
        help="Increase verbosity (-v, -vv, -vvv)",
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and exit without running",
    )
    
    parser.add_argument(
        "--init-config",
        action="store_true",
        help="Print default config YAML and exit",
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="CGW Coding Agent v2.0",
    )
    
    return parser


def setup_logging(verbosity: int) -> None:
    """Configure logging based on verbosity level."""
    if verbosity == 0:
        level = logging.INFO
    elif verbosity == 1:
        level = logging.DEBUG
    else:
        level = logging.DEBUG
        # Enable third-party debug logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    logging.getLogger("cgw_ssl_guard").setLevel(level)


def run_agent(args: argparse.Namespace) -> int:
    """Run the CGW coding agent."""
    from .config import CGWConfig, create_default_config
    from .coding_agent_runtime import CodingAgentRuntime, AgentConfig
    
    # Load config
    if args.config:
        config = CGWConfig.from_file(args.config)
        logger.info(f"Loaded config from {args.config}")
    else:
        config = CGWConfig()
    
    # Apply CLI overrides
    if args.goal:
        config.agent.goal = args.goal
    if args.repo:
        config.agent.repo_url = args.repo
    if args.branch:
        config.agent.repo_branch = args.branch
    if args.max_cycles:
        config.agent.max_cycles = args.max_cycles
    if args.timeout:
        config.agent.total_timeout = args.timeout
    if args.dashboard:
        config.dashboard.enabled = True
        config.dashboard.http_port = args.dashboard_port
    
    # Generate session ID
    session_id = args.session_id or f"cgw_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    config.agent.session_id = session_id
    logger.info(f"Session ID: {session_id}")
    
    # Create runtime with config
    agent_config = AgentConfig(
        max_cycles=config.agent.max_cycles,
        max_patches=config.agent.max_patches,
        max_test_runs=config.agent.max_test_runs,
        total_timeout=config.agent.total_timeout,
        goal=config.agent.goal,
    )
    
    runtime = CodingAgentRuntime(config=agent_config)
    
    # Start dashboard if enabled
    dashboard = None
    if config.dashboard.enabled:
        try:
            from .websocket_dashboard import CGWDashboardServer, DashboardConfig as DashConfig
            dash_config = DashConfig(
                http_port=config.dashboard.http_port,
                auto_open=config.dashboard.auto_open,
            )
            dashboard = CGWDashboardServer(dash_config)
            dashboard.start()
            logger.info(f"Dashboard: http://localhost:{config.dashboard.http_port}")
        except Exception as e:
            logger.warning(f"Failed to start dashboard: {e}")
    
    # Wire event store if enabled
    event_store = None
    if config.event_store.enabled:
        try:
            from .event_store import CGWEventStore, EventStoreSubscriber
            from .event_store import EventStoreConfig as ESConfig
            
            es_config = ESConfig(db_path=config.event_store.db_path)
            event_store = CGWEventStore(es_config)
            event_store.start_session(session_id, goal=config.agent.goal)
            
            # Subscribe to runtime events
            subscriber = EventStoreSubscriber(event_store, session_id)
            subscriber.subscribe(runtime.event_bus)
            logger.info(f"Event store: {config.event_store.db_path}")
        except Exception as e:
            logger.warning(f"Failed to initialize event store: {e}")
    
    # Wire bandit if enabled
    bandit = None
    if config.bandit.enabled:
        try:
            from .cgw_bandit import CGWBandit, CGWBanditConfig
            bandit_config = CGWBanditConfig(
                db_path=config.bandit.db_path,
                exploration_bonus=config.bandit.exploration_bonus,
            )
            bandit = CGWBandit(bandit_config)
            logger.info(f"Bandit: {config.bandit.db_path}")
        except Exception as e:
            logger.warning(f"Failed to initialize bandit: {e}")
    
    # Run the agent
    logger.info(f"Starting CGW agent: {config.agent.goal}")
    logger.info(f"Max cycles: {config.agent.max_cycles}")
    
    try:
        result = runtime.run_until_done()
        
        # Print summary
        print("\n" + "=" * 60)
        print(result.summary())
        print("=" * 60)
        
        # Export events if output specified
        if args.output and event_store:
            try:
                count = event_store.export_session_json(session_id, args.output)
                logger.info(f"Exported {count} events to {args.output}")
            except Exception as e:
                logger.error(f"Failed to export events: {e}")
        
        # End session
        if event_store:
            event_store.end_session(
                session_id,
                status="completed" if result.success else "failed",
                total_cycles=result.cycles_executed,
            )
        
        return 0 if result.success else 1
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"Agent failed: {e}")
        return 1
    finally:
        if dashboard:
            dashboard.stop()
        if event_store:
            event_store.close()
        if bandit:
            bandit.close()


def main(argv: Optional[list] = None) -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)
    
    # Handle --init-config
    if args.init_config:
        from .config import create_default_config
        print(create_default_config())
        return 0
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Handle --dry-run
    if args.dry_run:
        from .config import CGWConfig
        try:
            if args.config:
                config = CGWConfig.from_file(args.config)
            else:
                config = CGWConfig()
            
            print("Configuration valid:")
            print(f"  Goal: {config.agent.goal}")
            print(f"  Max cycles: {config.agent.max_cycles}")
            print(f"  LLM: {config.llm.provider}/{config.llm.model}")
            print(f"  Dashboard: {'enabled' if config.dashboard.enabled else 'disabled'}")
            return 0
        except Exception as e:
            print(f"Configuration error: {e}", file=sys.stderr)
            return 1
    
    # Run agent
    return run_agent(args)


if __name__ == "__main__":
    sys.exit(main())
