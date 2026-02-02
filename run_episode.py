#!/usr/bin/env python3
"""CLI for running orchestrated episodes with full stack integration.

This is the entry point for testing the complete system:
- LLM patch generation
- Staged test runner
- Failure triage
- Patch scoring/minimization

Usage:
    python run_episode.py --task-id django-12345 \\
        --problem "Fix AttributeError in dataclass" \\
        --repo-path /path/to/repo \\
        --test-command "pytest tests/" \\
        --llm-provider openai \\
        --llm-model gpt-4-turbo-preview

Example (Simulated):
    python run_episode.py --task-id test-001 \\
        --problem "Fix import error" \\
        --repo-path . \\
        --test-command "pytest tests/" \\
        --profile swebench_lite
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from agent.orchestrator import run_orchestrated_episode
from agent.types import Phase

try:
    from rfsn_controller.structured_logging import get_logger
except ImportError:
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    def get_logger(name):
        return logging.getLogger(name)

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run orchestrated episode with full stack integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Run with OpenAI GPT-4:
  python run_episode.py --task-id test-001 \\
      --problem "Fix AttributeError in dataclass init" \\
      --repo-path /tmp/myrepo \\
      --test-command "pytest tests/test_models.py" \\
      --llm-provider openai \\
      --llm-model gpt-4-turbo-preview

  # Run with Anthropic Claude:
  python run_episode.py --task-id test-002 \\
      --problem "Fix import error" \\
      --repo-path . \\
      --test-command "python -m pytest" \\
      --llm-provider anthropic \\
      --llm-model claude-3-5-sonnet-20241022

  # Run with DeepSeek:
  python run_episode.py --task-id test-003 \\
      --problem "Fix type error" \\
      --repo-path . \\
      --llm-provider deepseek \\
      --llm-model deepseek-chat

Environment Variables:
  OPENAI_API_KEY       - OpenAI API key
  ANTHROPIC_API_KEY    - Anthropic API key
  DEEPSEEK_API_KEY     - DeepSeek API key
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--task-id",
        required=True,
        help="Task identifier (e.g., django-12345)",
    )
    parser.add_argument(
        "--problem",
        required=True,
        help="Problem statement/description",
    )
    parser.add_argument(
        "--repo-path",
        required=True,
        help="Path to repository",
    )
    
    # Optional arguments
    parser.add_argument(
        "--test-command",
        default="pytest tests/",
        help="Test command to run (default: pytest tests/)",
    )
    parser.add_argument(
        "--profile",
        default="swebench_lite",
        choices=["swebench_lite", "swebench_verified"],
        help="Agent profile (default: swebench_lite)",
    )
    parser.add_argument(
        "--llm-provider",
        default="openai",
        choices=["openai", "anthropic", "deepseek"],
        help="LLM provider (default: openai)",
    )
    parser.add_argument(
        "--llm-model",
        help="LLM model name (default: provider-specific)",
    )
    parser.add_argument(
        "--use-docker",
        action="store_true",
        help="Use Docker for test isolation",
    )
    parser.add_argument(
        "--output",
        help="Output file for final state (JSON)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Validate repo path
    repo_path = Path(args.repo_path).resolve()
    if not repo_path.exists():
        logger.error(f"Repository path does not exist: {repo_path}")
        return 1
    
    # Check API keys
    if args.llm_provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY not set, LLM generation may fail")
    elif args.llm_provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        logger.warning("ANTHROPIC_API_KEY not set, LLM generation may fail")
    elif args.llm_provider == "deepseek" and not os.getenv("DEEPSEEK_API_KEY"):
        logger.warning("DEEPSEEK_API_KEY not set, LLM generation may fail")
    
    # Set default model if not specified
    llm_model = args.llm_model
    if not llm_model:
        default_models = {
            "openai": "gpt-4-turbo-preview",
            "anthropic": "claude-3-5-sonnet-20241022",
            "deepseek": "deepseek-chat",
        }
        llm_model = default_models[args.llm_provider]
    
    # Print configuration
    logger.info("=" * 80)
    logger.info("RFSN SWE-Bench Killer - Orchestrated Episode")
    logger.info("=" * 80)
    logger.info(f"Task ID: {args.task_id}")
    logger.info(f"Problem: {args.problem}")
    logger.info(f"Repository: {repo_path}")
    logger.info(f"Test Command: {args.test_command}")
    logger.info(f"Profile: {args.profile}")
    logger.info(f"LLM: {args.llm_provider}/{llm_model}")
    logger.info(f"Docker: {args.use_docker}")
    logger.info("=" * 80)
    
    try:
        # Run orchestrated episode
        logger.info("Starting orchestrated episode...")
        
        final_state = run_orchestrated_episode(
            task_id=args.task_id,
            problem_statement=args.problem,
            repo_path=str(repo_path),
            test_command=args.test_command,
            profile_name=args.profile,
            llm_provider=args.llm_provider,
            llm_model=llm_model,
            use_docker=args.use_docker,
        )
        
        # Print results
        logger.info("=" * 80)
        logger.info("Episode Complete")
        logger.info("=" * 80)
        logger.info(f"Final Phase: {final_state.phase.value}")
        logger.info(f"Rounds: {final_state.budget.round_idx}")
        logger.info(f"Patch Attempts: {final_state.budget.patch_attempts}")
        logger.info(f"Test Runs: {final_state.budget.test_runs}")
        logger.info(f"Model Calls: {final_state.budget.model_calls}")
        logger.info(f"Touched Files: {len(final_state.touched_files)}")
        logger.info(f"Stop Reason: {final_state.notes.get('stop_reason', 'unknown')}")
        
        # Check success
        success = final_state.phase == Phase.DONE and final_state.notes.get("stop_reason") == "finalized"
        
        if success:
            logger.info("✅ Episode completed successfully!")
        else:
            logger.warning(f"⚠️  Episode incomplete: {final_state.notes.get('stop_reason', 'unknown')}")
        
        # Save output if requested
        if args.output:
            output_data = {
                "task_id": final_state.task_id,
                "phase": final_state.phase.value,
                "rounds": final_state.budget.round_idx,
                "patch_attempts": final_state.budget.patch_attempts,
                "test_runs": final_state.budget.test_runs,
                "model_calls": final_state.budget.model_calls,
                "touched_files": final_state.touched_files,
                "stop_reason": final_state.notes.get("stop_reason", "unknown"),
                "success": success,
                "notes": final_state.notes,
            }
            
            output_path = Path(args.output)
            output_path.write_text(json.dumps(output_data, indent=2))
            logger.info(f"Results saved to: {output_path}")
        
        return 0 if success else 1
    
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        return 130
    
    except Exception as e:
        logger.error(f"Episode failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
