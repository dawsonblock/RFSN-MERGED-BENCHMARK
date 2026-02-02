"""Explicit planning layer with DAG-based execution.

This module implements a multi-step planner that:
1. Generates a DAG of execution nodes from a problem description
2. Executes nodes in topological order
3. Verifies each step before proceeding
4. Emits structured events for all actions
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .context import ControllerContext
    from .repo_index import RepoIndex


class NodeStatus(Enum):
    """Status of a plan node."""
    
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PlanNode:
    """A single node in the execution plan.
    
    Each node represents a discrete step with preconditions, actions,
    and verification criteria.
    """
    
    id: str
    description: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    preconditions: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    verification: str = ""
    status: NodeStatus = NodeStatus.PENDING
    result: dict[str, Any] | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert node to JSON-serializable dict."""
        return {
            "id": self.id,
            "description": self.description,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "preconditions": self.preconditions,
            "actions": self.actions,
            "verification": self.verification,
            "status": self.status.value,
            "result": self.result,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlanNode:
        """Create a node from a dictionary."""
        node = cls(
            id=data["id"],
            description=data["description"],
            inputs=data.get("inputs", []),
            outputs=data.get("outputs", []),
            preconditions=data.get("preconditions", []),
            actions=data.get("actions", []),
            verification=data.get("verification", ""),
        )
        if "status" in data:
            node.status = NodeStatus(data["status"])
        node.result = data.get("result")
        return node


@dataclass
class PlanDAG:
    """A directed acyclic graph of plan nodes.
    
    Supports topological sorting, cycle detection, and incremental execution.
    """
    
    nodes: dict[str, PlanNode] = field(default_factory=dict)
    edges: list[tuple[str, str]] = field(default_factory=list)
    
    def add_node(self, node: PlanNode) -> None:
        """Add a node to the graph."""
        self.nodes[node.id] = node
    
    def add_edge(self, from_id: str, to_id: str) -> None:
        """Add an edge from one node to another."""
        if from_id not in self.nodes or to_id not in self.nodes:
            raise ValueError(f"Both nodes must exist: {from_id} -> {to_id}")
        self.edges.append((from_id, to_id))
    
    def get_predecessors(self, node_id: str) -> list[str]:
        """Get all predecessor node IDs."""
        return [from_id for from_id, to_id in self.edges if to_id == node_id]
    
    def get_successors(self, node_id: str) -> list[str]:
        """Get all successor node IDs."""
        return [to_id for from_id, to_id in self.edges if from_id == node_id]
    
    def detect_cycles(self) -> bool:
        """Check if the graph contains cycles.
        
        Returns:
            True if cycles exist, False otherwise.
        """
        # Build adjacency list
        graph: dict[str, list[str]] = defaultdict(list)
        for from_id, to_id in self.edges:
            graph[from_id].append(to_id)
        
        # Track visited and recursion stack
        visited: set[str] = set()
        rec_stack: set[str] = set()
        
        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph[node]:
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node_id in self.nodes:
            if node_id not in visited:
                if dfs(node_id):
                    return True
        
        return False
    
    def topological_sort(self) -> list[str]:
        """Get nodes in topological order.
        
        Returns:
            List of node IDs in execution order.
            
        Raises:
            ValueError: If the graph contains cycles.
        """
        if self.detect_cycles():
            raise ValueError("Cannot sort: graph contains cycles")
        
        # Calculate in-degrees
        in_degree: dict[str, int] = {node_id: 0 for node_id in self.nodes}
        for _, to_id in self.edges:
            in_degree[to_id] += 1
        
        # Start with nodes that have no predecessors
        queue = deque([node_id for node_id, degree in in_degree.items() if degree == 0])
        result: list[str] = []
        
        while queue:
            node_id = queue.popleft()
            result.append(node_id)
            
            for successor in self.get_successors(node_id):
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)
        
        return result
    
    def get_ready_nodes(self) -> list[str]:
        """Get nodes that are ready to execute.
        
        A node is ready if all predecessors are DONE.
        
        Returns:
            List of node IDs ready for execution.
        """
        ready = []
        for node_id, node in self.nodes.items():
            if node.status != NodeStatus.PENDING:
                continue
            
            predecessors = self.get_predecessors(node_id)
            all_done = all(
                self.nodes[p].status == NodeStatus.DONE
                for p in predecessors
            )
            if all_done:
                ready.append(node_id)
        
        return ready
    
    def to_json(self) -> dict[str, Any]:
        """Convert DAG to JSON-serializable dict."""
        return {
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "edges": self.edges,
        }
    
    @classmethod
    def from_json(cls, data: dict[str, Any]) -> PlanDAG:
        """Create a DAG from JSON data."""
        dag = cls()
        for node_id, node_data in data.get("nodes", {}).items():
            dag.nodes[node_id] = PlanNode.from_dict(node_data)
        dag.edges = [tuple(e) for e in data.get("edges", [])]
        return dag


class Planner:
    """Generates and executes plans for code repair/generation tasks."""
    
    def __init__(self, context: ControllerContext) -> None:
        """Initialize the planner.
        
        Args:
            context: The controller context.
        """
        self.context = context
    
    def generate_plan(
        self,
        problem: str,
        repo_index: RepoIndex | None = None,
        mode: str = "repair",
    ) -> PlanDAG:
        """Generate an execution plan for the given problem.
        
        Args:
            problem: Description of the problem to solve.
            repo_index: Optional repository index for context.
            mode: Execution mode (repair, refactor, feature).
            
        Returns:
            A PlanDAG with execution nodes.
        """
        dag = PlanDAG()
        
        # Standard repair plan template
        if mode == "repair":
            # Node 1: Analyze failing tests
            dag.add_node(PlanNode(
                id="analyze",
                description="Analyze failing tests and identify root cause",
                outputs=["failure_analysis"],
                actions=["run_tests", "parse_errors", "identify_cause"],
                verification="failure analysis complete",
            ))
            
            # Node 2: Gather context
            dag.add_node(PlanNode(
                id="gather_context",
                description="Read relevant source files",
                inputs=["failure_analysis"],
                outputs=["source_context"],
                actions=["read_files", "trace_dependencies"],
                verification="source context gathered",
            ))
            dag.add_edge("analyze", "gather_context")
            
            # Node 3: Generate patch
            dag.add_node(PlanNode(
                id="generate_patch",
                description="Generate fix patch",
                inputs=["failure_analysis", "source_context"],
                outputs=["patch"],
                actions=["propose_patch", "validate_syntax"],
                verification="patch generated",
            ))
            dag.add_edge("gather_context", "generate_patch")
            
            # Node 4: Apply and verify
            dag.add_node(PlanNode(
                id="verify",
                description="Apply patch and run tests",
                inputs=["patch"],
                outputs=["verification_result"],
                actions=["apply_patch", "run_tests"],
                verification="all tests pass",
            ))
            dag.add_edge("generate_patch", "verify")
        
        elif mode == "feature":
            # Feature mode plan
            dag.add_node(PlanNode(
                id="understand",
                description="Understand feature requirements",
                outputs=["requirements"],
                actions=["parse_description", "identify_scope"],
            ))
            
            dag.add_node(PlanNode(
                id="design",
                description="Design solution approach",
                inputs=["requirements"],
                outputs=["design"],
                actions=["identify_files", "plan_changes"],
            ))
            dag.add_edge("understand", "design")
            
            dag.add_node(PlanNode(
                id="implement",
                description="Implement changes",
                inputs=["design"],
                outputs=["implementation"],
                actions=["write_code", "add_tests"],
            ))
            dag.add_edge("design", "implement")
            
            dag.add_node(PlanNode(
                id="verify",
                description="Verify implementation",
                inputs=["implementation"],
                outputs=["result"],
                actions=["run_tests", "lint"],
            ))
            dag.add_edge("implement", "verify")
        
        else:
            # Default simple plan
            dag.add_node(PlanNode(
                id="execute",
                description=f"Execute {mode} task",
                actions=["analyze", "act", "verify"],
            ))
        
        self.context.event_log.emit(
            "plan_generated",
            mode=mode,
            nodes=len(dag.nodes),
            edges=len(dag.edges),
        )
        
        return dag
    
    def execute_node(
        self,
        node: PlanNode,
        action_fn: Callable[[str], dict[str, Any]],
    ) -> bool:
        """Execute a single plan node.
        
        Args:
            node: The node to execute.
            action_fn: Function to execute each action.
            
        Returns:
            True if node completed successfully.
        """
        node.status = NodeStatus.RUNNING
        self.context.event_log.emit("plan_node_start", node_id=node.id)
        
        try:
            results = {}
            for action in node.actions:
                result = action_fn(action)
                results[action] = result
                
                if not result.get("ok", False):
                    node.status = NodeStatus.FAILED
                    node.result = results
                    self.context.event_log.emit(
                        "plan_node_fail",
                        node_id=node.id,
                        failed_action=action,
                    )
                    return False
            
            node.status = NodeStatus.DONE
            node.result = results
            self.context.event_log.emit("plan_node_done", node_id=node.id)
            return True
            
        except Exception as e:
            node.status = NodeStatus.FAILED
            node.result = {"error": str(e)}
            self.context.event_log.emit(
                "plan_node_fail",
                node_id=node.id,
                error=str(e),
            )
            return False
    
    def execute_plan(
        self,
        dag: PlanDAG,
        action_fn: Callable[[str], dict[str, Any]],
    ) -> bool:
        """Execute all nodes in the plan.
        
        Args:
            dag: The plan DAG to execute.
            action_fn: Function to execute each action.
            
        Returns:
            True if all nodes completed successfully.
        """
        order = dag.topological_sort()
        
        for node_id in order:
            node = dag.nodes[node_id]
            if not self.execute_node(node, action_fn):
                return False
        
        return True
