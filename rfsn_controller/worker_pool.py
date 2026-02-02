"""Multi-agent worker pool for parallel patch generation.
from __future__ import annotations

Implements:
- Stateless worker agents that propose patches
- WorkerPatchProposal schema for structured outputs
- Task DAG decomposition for complex fixes
- Conflict resolver for merging N candidate patches
"""

import hashlib
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Status of a task in the DAG."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class ConflictResolution(Enum):
    """Strategy for resolving patch conflicts."""
    FIRST_WINS = "first_wins"
    HIGHEST_CONFIDENCE = "highest_confidence"
    MERGE_SEQUENTIAL = "merge_sequential"
    VOTE = "vote"


@dataclass
class WorkerPatchProposal:
    """Structured output from a worker agent.
    
    Workers ONLY propose patches - they cannot write state or modify the repo.
    All proposals are collected and evaluated by the orchestrator.
    """
    
    # Identification
    worker_id: str
    task_id: str
    proposal_id: str
    
    # The patch
    diff: str
    confidence: float  # 0.0 - 1.0
    
    # Reasoning
    reasoning: str = ""
    approach: str = ""  # e.g., "surgical", "refactor", "workaround"
    
    # Metadata
    estimated_lines: int = 0
    estimated_files: int = 0
    dependencies: list[str] = field(default_factory=list)  # Other task_ids this depends on
    
    # Validation hints
    expected_tests_fixed: list[str] = field(default_factory=list)
    potential_regressions: list[str] = field(default_factory=list)
    
    def proposal_hash(self) -> str:
        """Hash for deduplication."""
        return hashlib.sha256(self.diff.encode()).hexdigest()[:16]
    
    def as_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "task_id": self.task_id,
            "proposal_id": self.proposal_id,
            "diff": self.diff,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "approach": self.approach,
            "estimated_lines": self.estimated_lines,
            "estimated_files": self.estimated_files,
            "dependencies": self.dependencies,
            "expected_tests_fixed": self.expected_tests_fixed,
            "potential_regressions": self.potential_regressions,
        }


@dataclass
class TaskNode:
    """A node in the task DAG."""
    
    task_id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    
    # DAG structure
    dependencies: list[str] = field(default_factory=list)  # task_ids that must complete first
    dependents: list[str] = field(default_factory=list)   # task_ids that depend on this
    
    # Execution info
    assigned_worker: str | None = None
    proposals: list[WorkerPatchProposal] = field(default_factory=list)
    selected_proposal: WorkerPatchProposal | None = None
    
    # Context for the worker
    context: dict[str, Any] = field(default_factory=dict)
    
    def is_ready(self, completed_tasks: set[str]) -> bool:
        """Check if all dependencies are satisfied."""
        return all(dep in completed_tasks for dep in self.dependencies)


class WorkerAgent(ABC):
    """Abstract base for stateless worker agents.
    
    Workers:
    - Receive a task description and context
    - Propose one or more patches
    - CANNOT write state or modify the repo directly
    """
    
    def __init__(self, worker_id: str):
        self.worker_id = worker_id
    
    @abstractmethod
    def generate_proposals(
        self,
        task: TaskNode,
        *,
        max_proposals: int = 3,
    ) -> list[WorkerPatchProposal]:
        """Generate patch proposals for a task.
        
        Args:
            task: The task to work on.
            max_proposals: Maximum number of proposals to generate.
        
        Returns:
            List of patch proposals (may be empty if no solution found).
        """
        pass


class SimpleWorkerAgent(WorkerAgent):
    """Simple worker that uses a patch generator function."""
    
    def __init__(
        self,
        worker_id: str,
        generator: Callable[[TaskNode], list[tuple[str, float]]],
    ):
        """Initialize with a generator function.
        
        Args:
            worker_id: Unique worker identifier.
            generator: Function that takes a task and returns [(diff, confidence), ...].
        """
        super().__init__(worker_id)
        self.generator = generator
    
    def generate_proposals(
        self,
        task: TaskNode,
        *,
        max_proposals: int = 3,
    ) -> list[WorkerPatchProposal]:
        try:
            results = self.generator(task)
        except Exception as e:
            logger.warning("Worker %s failed on task %s: %s", self.worker_id, task.task_id, e)
            return []
        
        proposals = []
        for i, (diff, confidence) in enumerate(results[:max_proposals]):
            proposal = WorkerPatchProposal(
                worker_id=self.worker_id,
                task_id=task.task_id,
                proposal_id=f"{self.worker_id}:{task.task_id}:{i}",
                diff=diff,
                confidence=confidence,
                estimated_lines=len(diff.splitlines()) if diff else 0,
            )
            proposals.append(proposal)
        
        return proposals


class TaskDAG:
    """Directed acyclic graph of tasks for complex repairs."""
    
    def __init__(self):
        self.tasks: dict[str, TaskNode] = {}
        self.completed: set[str] = set()
    
    def add_task(
        self,
        task_id: str,
        description: str,
        *,
        dependencies: list[str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> TaskNode:
        """Add a task to the DAG.
        
        Args:
            task_id: Unique task identifier.
            description: Task description for workers.
            dependencies: List of task_ids that must complete first.
            context: Additional context for the worker.
        
        Returns:
            The created TaskNode.
        """
        task = TaskNode(
            task_id=task_id,
            description=description,
            dependencies=dependencies or [],
            context=context or {},
        )
        
        # Update dependents of dependency tasks
        for dep_id in task.dependencies:
            if dep_id in self.tasks:
                self.tasks[dep_id].dependents.append(task_id)
        
        self.tasks[task_id] = task
        return task
    
    def get_ready_tasks(self) -> list[TaskNode]:
        """Get all tasks ready to execute (dependencies satisfied)."""
        ready = []
        for task in self.tasks.values():
            if task.status == TaskStatus.PENDING and task.is_ready(self.completed):
                ready.append(task)
        return ready
    
    def mark_completed(self, task_id: str, proposal: WorkerPatchProposal | None = None) -> None:
        """Mark a task as completed."""
        if task_id in self.tasks:
            self.tasks[task_id].status = TaskStatus.COMPLETED
            self.tasks[task_id].selected_proposal = proposal
            self.completed.add(task_id)
    
    def mark_failed(self, task_id: str) -> None:
        """Mark a task as failed."""
        if task_id in self.tasks:
            self.tasks[task_id].status = TaskStatus.FAILED
            # Mark dependents as blocked
            for dep_id in self.tasks[task_id].dependents:
                if dep_id in self.tasks:
                    self.tasks[dep_id].status = TaskStatus.BLOCKED
    
    def all_completed(self) -> bool:
        """Check if all tasks are completed or failed."""
        for task in self.tasks.values():
            if task.status in (TaskStatus.PENDING, TaskStatus.IN_PROGRESS):
                return False
        return True
    
    def get_execution_order(self) -> list[str]:
        """Get topological order of tasks."""
        # Kahn's algorithm
        in_degree = {tid: len(t.dependencies) for tid, t in self.tasks.items()}
        queue = [tid for tid, deg in in_degree.items() if deg == 0]
        order = []
        
        while queue:
            tid = queue.pop(0)
            order.append(tid)
            for dependent in self.tasks[tid].dependents:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        return order


class ConflictResolver:
    """Resolve conflicts between multiple patch proposals."""
    
    def __init__(self, strategy: ConflictResolution = ConflictResolution.HIGHEST_CONFIDENCE):
        self.strategy = strategy
    
    def resolve(
        self,
        proposals: list[WorkerPatchProposal],
        *,
        validation_fn: Callable[[str], bool] | None = None,
    ) -> WorkerPatchProposal | None:
        """Select the best proposal from candidates.
        
        Args:
            proposals: List of candidate proposals.
            validation_fn: Optional function to validate a diff.
        
        Returns:
            Selected proposal, or None if no valid proposal found.
        """
        if not proposals:
            return None
        
        # Filter out empty diffs
        valid = [p for p in proposals if p.diff and p.diff.strip()]
        if not valid:
            return None
        
        # Apply validation function if provided
        if validation_fn:
            valid = [p for p in valid if validation_fn(p.diff)]
            if not valid:
                return None
        
        if self.strategy == ConflictResolution.FIRST_WINS:
            return valid[0]
        
        elif self.strategy == ConflictResolution.HIGHEST_CONFIDENCE:
            return max(valid, key=lambda p: p.confidence)
        
        elif self.strategy == ConflictResolution.VOTE:
            # Group by diff hash and pick most common
            votes: dict[str, list[WorkerPatchProposal]] = {}
            for p in valid:
                h = p.proposal_hash()
                if h not in votes:
                    votes[h] = []
                votes[h].append(p)
            
            best_group = max(votes.values(), key=len)
            # Return highest confidence from best group
            return max(best_group, key=lambda p: p.confidence)
        
        elif self.strategy == ConflictResolution.MERGE_SEQUENTIAL:
            # Just return highest confidence; actual merging is complex
            logger.warning("MERGE_SEQUENTIAL not fully implemented, using HIGHEST_CONFIDENCE")
            return max(valid, key=lambda p: p.confidence)
        
        return valid[0]


class WorkerPool:
    """Pool of worker agents for parallel patch generation."""
    
    def __init__(
        self,
        workers: list[WorkerAgent] | None = None,
        conflict_resolver: ConflictResolver | None = None,
    ):
        """Initialize worker pool.
        
        Args:
            workers: List of worker agents.
            conflict_resolver: Resolver for conflicting proposals.
        """
        self.workers: dict[str, WorkerAgent] = {}
        if workers:
            for w in workers:
                self.workers[w.worker_id] = w
        
        self.resolver = conflict_resolver or ConflictResolver()
    
    def add_worker(self, worker: WorkerAgent) -> None:
        """Add a worker to the pool."""
        self.workers[worker.worker_id] = worker
    
    def remove_worker(self, worker_id: str) -> None:
        """Remove a worker from the pool."""
        self.workers.pop(worker_id, None)
    
    def process_task(
        self,
        task: TaskNode,
        *,
        max_proposals_per_worker: int = 2,
        validation_fn: Callable[[str], bool] | None = None,
    ) -> WorkerPatchProposal | None:
        """Process a task by collecting proposals from all workers.
        
        Args:
            task: The task to process.
            max_proposals_per_worker: Max proposals each worker can submit.
            validation_fn: Optional validation function for proposals.
        
        Returns:
            Selected proposal, or None if no valid proposal.
        """
        all_proposals: list[WorkerPatchProposal] = []
        
        for worker in self.workers.values():
            try:
                proposals = worker.generate_proposals(
                    task,
                    max_proposals=max_proposals_per_worker,
                )
                all_proposals.extend(proposals)
                logger.debug(
                    "Worker %s generated %d proposals for task %s",
                    worker.worker_id,
                    len(proposals),
                    task.task_id,
                )
            except Exception as e:
                logger.warning(
                    "Worker %s failed on task %s: %s",
                    worker.worker_id,
                    task.task_id,
                    e,
                )
        
        task.proposals = all_proposals
        
        # Resolve conflicts
        selected = self.resolver.resolve(all_proposals, validation_fn=validation_fn)
        if selected:
            logger.info(
                "Selected proposal %s (confidence=%.2f) for task %s",
                selected.proposal_id,
                selected.confidence,
                task.task_id,
            )
        
        return selected
    
    def process_dag(
        self,
        dag: TaskDAG,
        *,
        max_proposals_per_worker: int = 2,
        validation_fn: Callable[[str], bool] | None = None,
    ) -> dict[str, WorkerPatchProposal | None]:
        """Process all tasks in a DAG in dependency order.
        
        Args:
            dag: The task DAG to process.
            max_proposals_per_worker: Max proposals each worker can submit.
            validation_fn: Optional validation function.
        
        Returns:
            Dict of task_id -> selected proposal (or None).
        """
        results: dict[str, WorkerPatchProposal | None] = {}
        
        while not dag.all_completed():
            ready = dag.get_ready_tasks()
            if not ready:
                logger.warning("No ready tasks but DAG not complete - possible cycle")
                break
            
            for task in ready:
                task.status = TaskStatus.IN_PROGRESS
                proposal = self.process_task(
                    task,
                    max_proposals_per_worker=max_proposals_per_worker,
                    validation_fn=validation_fn,
                )
                
                if proposal:
                    dag.mark_completed(task.task_id, proposal)
                    results[task.task_id] = proposal
                else:
                    dag.mark_failed(task.task_id)
                    results[task.task_id] = None
        
        return results


def decompose_task(
    description: str,
    *,
    failing_tests: list[str],
    error_signatures: list[str],
) -> TaskDAG:
    """Decompose a complex repair into a DAG of subtasks.
    
    Args:
        description: High-level problem description.
        failing_tests: List of failing test names.
        error_signatures: List of error signatures.
    
    Returns:
        TaskDAG with decomposed subtasks.
    """
    dag = TaskDAG()
    
    # Group tests by file
    test_files: dict[str, list[str]] = {}
    for test in failing_tests:
        # Extract file from test path (e.g., "tests/test_foo.py::test_bar" -> "test_foo.py")
        if "::" in test:
            file_part = test.split("::")[0]
        else:
            file_part = test
        file_name = file_part.split("/")[-1] if "/" in file_part else file_part
        
        if file_name not in test_files:
            test_files[file_name] = []
        test_files[file_name].append(test)
    
    if len(test_files) == 1:
        # Simple case: all tests in one file, single task
        dag.add_task(
            task_id="fix_all",
            description=description,
            context={"failing_tests": failing_tests, "errors": error_signatures},
        )
    else:
        # Complex case: multiple files, create per-file subtasks
        subtask_ids = []
        for i, (file_name, tests) in enumerate(test_files.items()):
            task_id = f"fix_{file_name.replace('.', '_')}"
            dag.add_task(
                task_id=task_id,
                description=f"Fix tests in {file_name}: {', '.join(tests[:3])}...",
                context={"failing_tests": tests, "file": file_name},
            )
            subtask_ids.append(task_id)
        
        # Add integration task that depends on all subtasks
        dag.add_task(
            task_id="integrate",
            description="Verify all fixes work together",
            dependencies=subtask_ids,
            context={"all_failing_tests": failing_tests},
        )
    
    return dag
