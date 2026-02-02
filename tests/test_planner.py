"""Tests for the planner module."""

import pytest

from rfsn_controller.config import ControllerConfig
from rfsn_controller.context import create_context
from rfsn_controller.planner import NodeStatus, PlanDAG, Planner, PlanNode


class TestPlanNode:
    """Tests for PlanNode dataclass."""
    
    def test_create_node(self) -> None:
        """Test creating a plan node."""
        node = PlanNode(
            id="test",
            description="Test node",
            actions=["action1", "action2"],
        )
        
        assert node.id == "test"
        assert node.status == NodeStatus.PENDING
        assert len(node.actions) == 2
    
    def test_to_dict_and_back(self) -> None:
        """Test node serialization."""
        node = PlanNode(
            id="test",
            description="Test node",
            inputs=["input1"],
            outputs=["output1"],
            actions=["action1"],
            verification="check",
        )
        
        d = node.to_dict()
        restored = PlanNode.from_dict(d)
        
        assert restored.id == node.id
        assert restored.description == node.description
        assert restored.inputs == node.inputs


class TestPlanDAG:
    """Tests for PlanDAG class."""
    
    def test_add_nodes_and_edges(self) -> None:
        """Test adding nodes and edges."""
        dag = PlanDAG()
        
        dag.add_node(PlanNode(id="a", description="Node A"))
        dag.add_node(PlanNode(id="b", description="Node B"))
        dag.add_edge("a", "b")
        
        assert len(dag.nodes) == 2
        assert len(dag.edges) == 1
    
    def test_topological_sort_simple(self) -> None:
        """Test topological sort with simple graph."""
        dag = PlanDAG()
        
        dag.add_node(PlanNode(id="a", description="First"))
        dag.add_node(PlanNode(id="b", description="Second"))
        dag.add_node(PlanNode(id="c", description="Third"))
        
        dag.add_edge("a", "b")
        dag.add_edge("b", "c")
        
        order = dag.topological_sort()
        
        assert order.index("a") < order.index("b")
        assert order.index("b") < order.index("c")
    
    def test_topological_sort_parallel(self) -> None:
        """Test topological sort with parallel paths."""
        dag = PlanDAG()
        
        # Diamond shape: a -> b, a -> c, b -> d, c -> d
        dag.add_node(PlanNode(id="a", description="Start"))
        dag.add_node(PlanNode(id="b", description="Path 1"))
        dag.add_node(PlanNode(id="c", description="Path 2"))
        dag.add_node(PlanNode(id="d", description="End"))
        
        dag.add_edge("a", "b")
        dag.add_edge("a", "c")
        dag.add_edge("b", "d")
        dag.add_edge("c", "d")
        
        order = dag.topological_sort()
        
        assert order[0] == "a"
        assert order[-1] == "d"
        assert order.index("a") < order.index("b")
        assert order.index("a") < order.index("c")
    
    def test_detect_cycles_no_cycle(self) -> None:
        """Test cycle detection with acyclic graph."""
        dag = PlanDAG()
        
        dag.add_node(PlanNode(id="a", description="A"))
        dag.add_node(PlanNode(id="b", description="B"))
        dag.add_edge("a", "b")
        
        assert dag.detect_cycles() is False
    
    def test_detect_cycles_with_cycle(self) -> None:
        """Test cycle detection with cyclic graph."""
        dag = PlanDAG()
        
        dag.add_node(PlanNode(id="a", description="A"))
        dag.add_node(PlanNode(id="b", description="B"))
        dag.add_node(PlanNode(id="c", description="C"))
        
        dag.add_edge("a", "b")
        dag.add_edge("b", "c")
        dag.add_edge("c", "a")  # Creates cycle
        
        assert dag.detect_cycles() is True
    
    def test_topological_sort_rejects_cycles(self) -> None:
        """Test that topological sort raises on cycles."""
        dag = PlanDAG()
        
        dag.add_node(PlanNode(id="a", description="A"))
        dag.add_node(PlanNode(id="b", description="B"))
        dag.add_edge("a", "b")
        dag.add_edge("b", "a")
        
        with pytest.raises(ValueError, match="cycles"):
            dag.topological_sort()
    
    def test_get_ready_nodes(self) -> None:
        """Test finding ready nodes."""
        dag = PlanDAG()
        
        node_a = PlanNode(id="a", description="A")
        node_b = PlanNode(id="b", description="B")
        node_c = PlanNode(id="c", description="C")
        
        dag.add_node(node_a)
        dag.add_node(node_b)
        dag.add_node(node_c)
        dag.add_edge("a", "b")
        dag.add_edge("b", "c")
        
        # Initially only A is ready
        ready = dag.get_ready_nodes()
        assert ready == ["a"]
        
        # After A is done, B is ready
        node_a.status = NodeStatus.DONE
        ready = dag.get_ready_nodes()
        assert ready == ["b"]
    
    def test_to_json_and_back(self) -> None:
        """Test DAG serialization."""
        dag = PlanDAG()
        dag.add_node(PlanNode(id="a", description="A"))
        dag.add_node(PlanNode(id="b", description="B"))
        dag.add_edge("a", "b")
        
        json_data = dag.to_json()
        restored = PlanDAG.from_json(json_data)
        
        assert len(restored.nodes) == len(dag.nodes)
        assert len(restored.edges) == len(dag.edges)


class TestPlanner:
    """Tests for Planner class."""
    
    @pytest.fixture
    def context(self, tmp_path):
        """Create a test context."""
        config = ControllerConfig(
            github_url="https://github.com/test/repo",
            output_dir=str(tmp_path / ".rfsn"),
        )
        return create_context(config)
    
    def test_generate_repair_plan(self, context) -> None:
        """Test generating a repair mode plan."""
        planner = Planner(context)
        dag = planner.generate_plan("Fix failing test", mode="repair")
        
        assert len(dag.nodes) > 0
        assert "analyze" in dag.nodes
        assert "verify" in dag.nodes
    
    def test_generate_feature_plan(self, context) -> None:
        """Test generating a feature mode plan."""
        planner = Planner(context)
        dag = planner.generate_plan("Add authentication", mode="feature")
        
        assert len(dag.nodes) > 0
        assert "understand" in dag.nodes
        assert "implement" in dag.nodes
    
    def test_execute_node_success(self, context) -> None:
        """Test executing a successful node."""
        planner = Planner(context)
        node = PlanNode(id="test", description="Test", actions=["action1"])
        
        def action_fn(action: str):
            return {"ok": True}
        
        result = planner.execute_node(node, action_fn)
        
        assert result is True
        assert node.status == NodeStatus.DONE
    
    def test_execute_node_failure(self, context) -> None:
        """Test executing a failing node."""
        planner = Planner(context)
        node = PlanNode(id="test", description="Test", actions=["action1"])
        
        def action_fn(action: str):
            return {"ok": False, "error": "Failed"}
        
        result = planner.execute_node(node, action_fn)
        
        assert result is False
        assert node.status == NodeStatus.FAILED
    
    def test_execute_plan_all_success(self, context) -> None:
        """Test executing a complete plan successfully."""
        planner = Planner(context)
        dag = PlanDAG()
        
        dag.add_node(PlanNode(id="a", description="A", actions=["a1"]))
        dag.add_node(PlanNode(id="b", description="B", actions=["b1"]))
        dag.add_edge("a", "b")
        
        def action_fn(action: str):
            return {"ok": True}
        
        result = planner.execute_plan(dag, action_fn)
        
        assert result is True
        assert dag.nodes["a"].status == NodeStatus.DONE
        assert dag.nodes["b"].status == NodeStatus.DONE
