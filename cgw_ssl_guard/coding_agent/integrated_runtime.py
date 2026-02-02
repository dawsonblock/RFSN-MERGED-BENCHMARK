"""Integrated CGW Runtime with Phase 2 Components.

This module provides an enhanced runtime that wires together all Phase 2
components:
- Strategy Bandit: Adaptive action selection
- Event Store: Persistent event logging
- Action Memory: Similarity boosting and regression firewall
- Dashboard: Real-time WebSocket updates
- Streaming LLM: Token-by-token generation

Usage:
    from cgw_ssl_guard.coding_agent.integrated_runtime import IntegratedCGWAgent
    
    agent = IntegratedCGWAgent.from_config("cgw.yaml")
    result = agent.run()
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from .action_types import ActionPayload, CodingAction, CycleResult, ExecutionResult
from .coding_agent_runtime import CodingAgentRuntime, AgentConfig, AgentResult
from .config import CGWConfig, load_config

logger = logging.getLogger(__name__)


@dataclass
class IntegrationState:
    """State tracking for integrated components."""
    
    session_id: str = ""
    start_time: float = 0.0
    
    # Component references
    bandit: Any = None
    memory: Any = None
    event_store: Any = None
    dashboard: Any = None
    
    # Metrics
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    blocked_actions: int = 0
    boosted_proposals: int = 0


class IntegratedCGWAgent:
    """Fully integrated CGW Coding Agent with all Phase 2 features.
    
    This class wires together:
    - CodingAgentRuntime: Core serial decision loop
    - CGWBandit: Adaptive strategy selection
    - CGWEventStore: Persistent event logging
    - CGWActionMemory: Similarity-based proposal boosting
    - CGWDashboardServer: Real-time monitoring
    
    Example:
        agent = IntegratedCGWAgent(goal="Fix failing tests")
        result = agent.run()
    """
    
    def __init__(
        self,
        config: Optional[CGWConfig] = None,
        goal: str = "Fix failing tests",
        session_id: Optional[str] = None,
    ):
        self.config = config or CGWConfig()
        self.config.agent.goal = goal
        
        # Generate session ID
        self.session_id = session_id or f"cgw_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        # Integration state
        self._state = IntegrationState(
            session_id=self.session_id,
            start_time=time.time(),
        )
        
        # Core runtime (created lazily)
        self._runtime: Optional[CodingAgentRuntime] = None
        
        # Callbacks for custom integrations
        self._on_cycle: List[Callable[[CycleResult], None]] = []
        self._on_proposal: List[Callable[[ActionPayload, float], None]] = []
    
    @classmethod
    def from_config(
        cls,
        config_path: Union[str, Path],
        **overrides,
    ) -> "IntegratedCGWAgent":
        """Create agent from config file."""
        config = load_config(config_path)
        
        # Apply overrides
        for key, value in overrides.items():
            if hasattr(config.agent, key):
                setattr(config.agent, key, value)
        
        return cls(
            config=config,
            goal=config.agent.goal,
            session_id=config.agent.session_id,
        )
    
    def _init_runtime(self) -> CodingAgentRuntime:
        """Initialize the core runtime with integration hooks."""
        agent_config = AgentConfig(
            max_cycles=self.config.agent.max_cycles,
            max_patches=self.config.agent.max_patches,
            max_test_runs=self.config.agent.max_test_runs,
            total_timeout=self.config.agent.total_timeout,
            goal=self.config.agent.goal,
        )
        
        runtime = CodingAgentRuntime(config=agent_config)
        
        # Register cycle callback
        runtime.on_cycle_complete(self._on_cycle_complete)
        
        return runtime
    
    def _init_bandit(self) -> Any:
        """Initialize the strategy bandit."""
        if not self.config.bandit.enabled:
            return None
        
        try:
            from .cgw_bandit import CGWBandit, CGWBanditConfig
            
            bandit_config = CGWBanditConfig(
                db_path=self.config.bandit.db_path,
                exploration_bonus=self.config.bandit.exploration_bonus,
                decay_factor=self.config.bandit.decay_factor,
            )
            bandit = CGWBandit(bandit_config)
            logger.info(f"Bandit initialized: {self.config.bandit.db_path}")
            return bandit
        except Exception as e:
            logger.warning(f"Failed to initialize bandit: {e}")
            return None
    
    def _init_memory(self) -> Any:
        """Initialize action outcome memory."""
        if not self.config.memory.enabled:
            return None
        
        try:
            from .action_memory import CGWActionMemory, CGWMemoryConfig
            
            memory_config = CGWMemoryConfig(
                db_path=self.config.memory.db_path,
                half_life_days=self.config.memory.half_life_days,
                regression_threshold=self.config.memory.regression_threshold,
            )
            memory = CGWActionMemory(memory_config)
            memory.set_session(self.session_id)
            logger.info(f"Memory initialized: {self.config.memory.db_path}")
            return memory
        except Exception as e:
            logger.warning(f"Failed to initialize memory: {e}")
            return None
    
    def _init_event_store(self) -> Any:
        """Initialize event store."""
        if not self.config.event_store.enabled:
            return None
        
        try:
            from .event_store import CGWEventStore, EventStoreConfig, EventStoreSubscriber
            
            es_config = EventStoreConfig(db_path=self.config.event_store.db_path)
            event_store = CGWEventStore(es_config)
            event_store.start_session(self.session_id, goal=self.config.agent.goal)
            
            # Subscribe to runtime events
            if self._runtime:
                subscriber = EventStoreSubscriber(event_store, self.session_id)
                subscriber.subscribe(self._runtime.event_bus)
            
            logger.info(f"Event store initialized: {self.config.event_store.db_path}")
            return event_store
        except Exception as e:
            logger.warning(f"Failed to initialize event store: {e}")
            return None
    
    def _init_dashboard(self) -> Any:
        """Initialize dashboard server."""
        if not self.config.dashboard.enabled:
            return None
        
        try:
            from .websocket_dashboard import CGWDashboardServer, DashboardConfig, DashboardEventSubscriber
            
            dash_config = DashboardConfig(
                http_port=self.config.dashboard.http_port,
                ws_port=self.config.dashboard.ws_port,
                auto_open=self.config.dashboard.auto_open,
            )
            dashboard = CGWDashboardServer(dash_config)
            dashboard.start()
            
            # Subscribe to runtime events
            if self._runtime:
                subscriber = DashboardEventSubscriber(dashboard, self.session_id)
                subscriber.subscribe(self._runtime.event_bus)
            
            logger.info(f"Dashboard: http://localhost:{self.config.dashboard.http_port}")
            return dashboard
        except Exception as e:
            logger.warning(f"Failed to initialize dashboard: {e}")
            return None
    
    def _on_cycle_complete(self, result: CycleResult) -> None:
        """Handle cycle completion with integration updates."""
        # Update bandit with outcome
        if self._state.bandit and result.execution_result:
            success = result.execution_result.success
            reward = 1.0 if success else 0.0
            self._state.bandit.update(result.action.value, reward)
        
        # Record to memory
        if self._state.memory and result.execution_result:
            self._state.memory.record_outcome(
                action_type=result.action.value,
                action_key=result.slot_id or str(result.cycle_id),
                outcome="success" if result.execution_result.success else "failure",
                exec_time_ms=int(result.execution_time_ms),
            )
        
        # Emit to dashboard
        if self._state.dashboard:
            self._state.dashboard.emit_event({
                "event_type": "CYCLE_COMPLETE",
                "session_id": self.session_id,
                "cycle_id": result.cycle_id,
                "action": result.action.value,
                "success": result.execution_result.success if result.execution_result else False,
                "execution_time_ms": result.execution_time_ms,
            })
        
        # Call user callbacks
        for callback in self._on_cycle:
            try:
                callback(result)
            except Exception as e:
                logger.warning(f"Cycle callback error: {e}")
    
    def get_saliency_boost(self, action_type: str) -> float:
        """Get combined saliency boost from bandit and memory."""
        boost = 1.0
        
        if self._state.bandit:
            boost *= self._state.bandit.get_saliency_boost(action_type)
        
        if self._state.memory:
            boost *= self._state.memory.get_saliency_boost(action_type)
        
        if boost > 1.0:
            self._state.boosted_proposals += 1
        
        return boost
    
    def is_action_blocked(self, action_type: str, action_key: str) -> bool:
        """Check if action is blocked by regression firewall."""
        if self._state.memory:
            if self._state.memory.is_blocked(action_type, action_key):
                self._state.blocked_actions += 1
                return True
        return False
    
    def on_cycle(self, callback: Callable[[CycleResult], None]) -> None:
        """Register cycle completion callback."""
        self._on_cycle.append(callback)
    
    def run(self, max_cycles: Optional[int] = None) -> AgentResult:
        """Run the integrated agent.
        
        Args:
            max_cycles: Override for max cycles
            
        Returns:
            AgentResult with success status and history
        """
        logger.info(f"Starting integrated CGW agent: {self.session_id}")
        logger.info(f"Goal: {self.config.agent.goal}")
        
        try:
            # Initialize all components
            self._runtime = self._init_runtime()
            self._state.bandit = self._init_bandit()
            self._state.memory = self._init_memory()
            self._state.event_store = self._init_event_store()
            self._state.dashboard = self._init_dashboard()
            
            # Run the agent
            result = self._runtime.run_until_done(max_cycles=max_cycles)
            
            # Log summary
            logger.info(result.summary())
            
            # Record final metrics
            if self._state.dashboard:
                self._state.dashboard.emit_event({
                    "event_type": "AGENT_COMPLETE",
                    "session_id": self.session_id,
                    "success": result.success,
                    "cycles": result.cycles_executed,
                    "blocked_actions": self._state.blocked_actions,
                    "boosted_proposals": self._state.boosted_proposals,
                })
            
            return result
            
        finally:
            self._cleanup()
    
    def _cleanup(self) -> None:
        """Clean up all resources."""
        if self._state.event_store:
            try:
                self._state.event_store.end_session(
                    self.session_id,
                    status="completed",
                    total_cycles=len(self._runtime._cycle_history) if self._runtime else 0,
                )
                self._state.event_store.close()
            except Exception:
                pass
        
        if self._state.bandit:
            try:
                self._state.bandit.close()
            except Exception:
                pass
        
        if self._state.memory:
            try:
                self._state.memory.close()
            except Exception:
                pass
        
        if self._state.dashboard:
            try:
                self._state.dashboard.stop()
            except Exception:
                pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get integration statistics."""
        stats = {
            "session_id": self.session_id,
            "runtime_seconds": time.time() - self._state.start_time,
            "blocked_actions": self._state.blocked_actions,
            "boosted_proposals": self._state.boosted_proposals,
        }
        
        if self._state.bandit:
            stats["bandit"] = self._state.bandit.get_stats()
        
        if self._state.memory:
            stats["memory"] = self._state.memory.get_stats()
        
        if self._state.event_store:
            stats["event_store"] = self._state.event_store.get_stats()
        
        return stats


def run_agent(
    goal: str = "Fix failing tests",
    config_path: Optional[str] = None,
    **kwargs,
) -> AgentResult:
    """Convenience function to run the integrated agent.
    
    Args:
        goal: Goal description
        config_path: Optional config file path
        **kwargs: Additional config overrides
        
    Returns:
        AgentResult
    """
    if config_path:
        agent = IntegratedCGWAgent.from_config(config_path, goal=goal, **kwargs)
    else:
        agent = IntegratedCGWAgent(goal=goal)
    
    return agent.run()
