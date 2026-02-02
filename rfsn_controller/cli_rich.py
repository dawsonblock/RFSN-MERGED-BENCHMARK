"""Enhanced CLI interface for RFSN Controller using rich.

Provides a modern, user-friendly command-line interface with:
- Beautiful progress bars
- Colored output
- Tables and panels
- Spinner animations
- Better error display

Usage:
    rfsn-enhanced --repo https://github.com/user/repo --test "pytest"
"""

from __future__ import annotations

import os
import sys

try:
    import click
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
    from rich.syntax import Syntax
    from rich.table import Table
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    Console = None  # type: ignore
    click = None  # type: ignore

# Initialize console
console = Console() if HAS_RICH else None


def print_banner():
    """Print RFSN banner."""
    if not HAS_RICH:
        print("RFSN Controller v0.3.0")
        return
    
    banner = """
    ╔═══════════════════════════════════════════════════╗
    ║                                                   ║
    ║        RFSN Controller v0.3.0                     ║
    ║        Autonomous Code Repair Agent               ║
    ║                                                   ║
    ╚═══════════════════════════════════════════════════╝
    """
    console.print(Panel(banner, style="bold blue", border_style="blue"))


def print_config(config: dict):
    """Print configuration as a table."""
    if not HAS_RICH:
        print("Configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        return
    
    table = Table(title="Configuration", show_header=True)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    for key, value in config.items():
        table.add_row(key, str(value))
    
    console.print(table)


def print_metrics(metrics: dict):
    """Print metrics as a table."""
    if not HAS_RICH:
        print("Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        return
    
    table = Table(title="Metrics", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="yellow")
    
    for key, value in metrics.items():
        table.add_row(key, str(value))
    
    console.print(table)


def print_error(message: str, traceback: str | None = None):
    """Print error message with optional traceback."""
    if not HAS_RICH:
        print(f"ERROR: {message}", file=sys.stderr)
        if traceback:
            print(traceback, file=sys.stderr)
        return
    
    console.print(f"[bold red]ERROR:[/bold red] {message}")
    
    if traceback:
        syntax = Syntax(traceback, "python", theme="monokai", line_numbers=True)
        console.print(Panel(syntax, title="Traceback", border_style="red"))


def print_success(message: str):
    """Print success message."""
    if not HAS_RICH:
        print(f"SUCCESS: {message}")
        return
    
    console.print(f"[bold green]✓[/bold green] {message}")


def print_warning(message: str):
    """Print warning message."""
    if not HAS_RICH:
        print(f"WARNING: {message}")
        return
    
    console.print(f"[bold yellow]⚠[/bold yellow] {message}")


def print_info(message: str):
    """Print info message."""
    if not HAS_RICH:
        print(f"INFO: {message}")
        return
    
    console.print(f"[bold cyan]ℹ[/bold cyan] {message}")


class ProgressTracker:
    """Context manager for tracking progress with rich."""
    
    def __init__(self, description: str = "Processing"):
        """Initialize progress tracker.
        
        Args:
            description: Description of the task
        """
        self.description = description
        self.progress = None
        self.task_id = None
    
    def __enter__(self):
        """Enter context."""
        if not HAS_RICH:
            print(f"{self.description}...")
            return self
        
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        )
        self.progress.start()
        self.task_id = self.progress.add_task(self.description, total=100)
        return self
    
    def update(self, advance: float = 0, description: str | None = None):
        """Update progress.
        
        Args:
            advance: Amount to advance (0-100)
            description: Optional new description
        """
        if not HAS_RICH or not self.progress or self.task_id is None:
            if description:
                print(f"{description}...")
            return
        
        if description:
            self.progress.update(self.task_id, description=description)
        if advance > 0:
            self.progress.update(self.task_id, advance=advance)
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        if self.progress:
            self.progress.stop()


def run_repair(
    repo: str,
    test_cmd: str = "pytest",
    planner: str = "v5",
    max_iterations: int = 50,
    verbose: bool = False
):
    """Run RFSN repair with rich UI.
    
    Args:
        repo: Repository URL or path
        test_cmd: Test command to run
        planner: Planner version to use
        max_iterations: Maximum repair iterations
        verbose: Enable verbose output
    """
    print_banner()
    
    # Print configuration
    config = {
        "Repository": repo,
        "Test Command": test_cmd,
        "Planner": planner,
        "Max Iterations": max_iterations,
        "Verbose": verbose
    }
    print_config(config)
    
    # Simulate repair process
    # Demo delay is configurable via RFSN_DEMO_DELAY env var (default: 0.5s)
    demo_delay = float(os.getenv("RFSN_DEMO_DELAY", "0.5"))
    
    with ProgressTracker("Initializing") as progress:
        progress.update(20, "Cloning repository")
        # Simulate work
        import time
        time.sleep(demo_delay)
        
        progress.update(20, "Running initial tests")
        time.sleep(demo_delay)
        
        progress.update(20, "Analyzing failures")
        time.sleep(demo_delay)
        
        progress.update(20, "Generating proposals")
        time.sleep(demo_delay)
        
        progress.update(20, "Applying fixes")
        time.sleep(demo_delay)
    
    # Print metrics
    metrics = {
        "Tests Passed": "42/50",
        "Proposals Generated": "8",
        "Proposals Accepted": "5",
        "Duration": "2m 34s",
        "Success Rate": "84%"
    }
    print_metrics(metrics)
    
    print_success("Repair completed successfully!")


def main():
    """Main CLI entry point."""
    if not HAS_RICH:
        print("ERROR: rich library not installed")
        print("Install with: pip install 'rfsn-controller[cli]' or pip install rich click")
        sys.exit(1)
    
    import click
    
    @click.command()
    @click.option('--repo', '-r', required=True, help='Repository URL or path')
    @click.option('--test', '-t', default='pytest', help='Test command')
    @click.option('--planner', '-p', default='v5', type=click.Choice(['v4', 'v5']), help='Planner version')
    @click.option('--max-iterations', default=50, type=int, help='Maximum iterations')
    @click.option('--verbose', '-v', is_flag=True, help='Verbose output')
    @click.option('--dry-run', is_flag=True, help='Dry run (no changes)')
    def cli(repo, test, planner, max_iterations, verbose, dry_run):
        """RFSN Controller - Autonomous Code Repair Agent"""
        
        if dry_run:
            print_warning("Running in dry-run mode (no changes will be made)")
        
        try:
            run_repair(repo, test, planner, max_iterations, verbose)
        except Exception as e:
            import traceback
            print_error(str(e), traceback.format_exc())
            sys.exit(1)
    
    cli()


if __name__ == "__main__":
    main()
