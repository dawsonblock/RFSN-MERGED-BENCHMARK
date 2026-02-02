"""CGW Coding Agent Configuration.

Unified configuration system supporting:
- YAML/JSON config files
- Environment variable overrides
- Dataclass-based schema with validation
- Default values and merging

Example config.yaml:
    agent:
      max_cycles: 100
      goal: "Fix failing tests"
    
    llm:
      provider: deepseek
      model: deepseek-coder
      temperature: 0.2
    
    sandbox:
      image: python:3.11-slim
      timeout: 300
    
    dashboard:
      enabled: true
      port: 8765
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def _load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Load YAML file, falling back to JSON."""
    path = Path(path)
    if not path.exists():
        return {}
    
    content = path.read_text()
    
    try:
        import yaml
        return yaml.safe_load(content) or {}
    except ImportError:
        # Fall back to JSON
        if path.suffix in ('.yaml', '.yml'):
            logger.warning("PyYAML not installed, trying JSON parse")
        return json.loads(content)


def _env_override(key: str, default: Any) -> Any:
    """Get environment variable with type coercion."""
    env_key = f"CGW_{key.upper()}"
    value = os.environ.get(env_key)
    
    if value is None:
        return default
    
    # Type coercion based on default type
    if isinstance(default, bool):
        return value.lower() in ('true', '1', 'yes')
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return value


@dataclass
class LLMConfig:
    """LLM provider configuration."""
    
    provider: str = "deepseek"
    model: str = "deepseek-coder"
    temperature: float = 0.2
    max_tokens: int = 4096
    timeout: float = 120.0
    
    # API keys (from env)
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    
    # Fallback chain
    fallback_providers: List[str] = field(default_factory=lambda: ["openai", "gemini"])
    
    def __post_init__(self):
        # Override from environment
        self.provider = _env_override("LLM_PROVIDER", self.provider)
        self.model = _env_override("LLM_MODEL", self.model)
        self.temperature = _env_override("LLM_TEMPERATURE", self.temperature)


@dataclass
class SandboxConfig:
    """Docker sandbox configuration."""
    
    image: str = "python:3.11-slim"
    timeout: int = 300
    memory_limit: str = "1g"
    cpu_limit: float = 1.0
    network_disabled: bool = False
    
    # Mount paths
    work_dir: str = "/workspace"
    
    def __post_init__(self):
        self.image = _env_override("SANDBOX_IMAGE", self.image)
        self.timeout = _env_override("SANDBOX_TIMEOUT", self.timeout)


@dataclass
class BanditConfig:
    """Strategy bandit configuration."""
    
    enabled: bool = True
    db_path: Optional[str] = None
    exploration_bonus: float = 0.1
    decay_factor: float = 0.99
    
    def __post_init__(self):
        if self.db_path is None:
            home = os.path.expanduser("~")
            self.db_path = os.path.join(home, ".cgw", "bandit.db")


@dataclass
class MemoryConfig:
    """Action outcome memory configuration."""
    
    enabled: bool = True
    db_path: Optional[str] = None
    half_life_days: int = 14
    regression_threshold: float = 0.2
    
    def __post_init__(self):
        if self.db_path is None:
            home = os.path.expanduser("~")
            self.db_path = os.path.join(home, ".cgw", "memory.db")


@dataclass
class EventStoreConfig:
    """Event store configuration."""
    
    enabled: bool = True
    db_path: Optional[str] = None
    max_events_per_session: int = 10000
    retention_days: int = 30
    
    def __post_init__(self):
        if self.db_path is None:
            home = os.path.expanduser("~")
            self.db_path = os.path.join(home, ".cgw", "events.db")


@dataclass
class DashboardConfig:
    """Dashboard configuration."""
    
    enabled: bool = True
    http_port: int = 8765
    ws_port: int = 8766
    auto_open: bool = True
    
    def __post_init__(self):
        self.http_port = _env_override("DASHBOARD_PORT", self.http_port)
        self.auto_open = _env_override("DASHBOARD_AUTO_OPEN", self.auto_open)


@dataclass 
class AgentConfig:
    """Main agent configuration."""
    
    # Core settings
    max_cycles: int = 100
    max_patches: int = 10
    max_test_runs: int = 20
    cycle_timeout: float = 600.0
    total_timeout: float = 3600.0
    
    # Goal
    goal: str = "Fix failing tests"
    
    # Repository
    repo_url: Optional[str] = None
    repo_branch: str = "main"
    
    # Session
    session_id: Optional[str] = None
    
    def __post_init__(self):
        self.max_cycles = _env_override("MAX_CYCLES", self.max_cycles)
        self.goal = _env_override("GOAL", self.goal)


@dataclass
class CGWConfig:
    """Complete CGW configuration schema."""
    
    agent: AgentConfig = field(default_factory=AgentConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    bandit: BanditConfig = field(default_factory=BanditConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    event_store: EventStoreConfig = field(default_factory=EventStoreConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CGWConfig":
        """Create config from dictionary."""
        return cls(
            agent=AgentConfig(**data.get("agent", {})),
            llm=LLMConfig(**data.get("llm", {})),
            sandbox=SandboxConfig(**data.get("sandbox", {})),
            bandit=BanditConfig(**data.get("bandit", {})),
            memory=MemoryConfig(**data.get("memory", {})),
            event_store=EventStoreConfig(**data.get("event_store", {})),
            dashboard=DashboardConfig(**data.get("dashboard", {})),
        )
    
    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "CGWConfig":
        """Load config from YAML/JSON file."""
        data = _load_yaml(path)
        return cls.from_dict(data)
    
    @classmethod
    def load(cls, path: Optional[Union[str, Path]] = None) -> "CGWConfig":
        """Load config from file or defaults.
        
        Search order:
        1. Explicit path if provided
        2. CGW_CONFIG environment variable
        3. ./cgw.yaml or ./cgw.json
        4. ~/.cgw/config.yaml
        5. Default config
        """
        # Explicit path
        if path:
            return cls.from_file(path)
        
        # Environment variable
        env_path = os.environ.get("CGW_CONFIG")
        if env_path and Path(env_path).exists():
            return cls.from_file(env_path)
        
        # Current directory
        for name in ["cgw.yaml", "cgw.yml", "cgw.json"]:
            if Path(name).exists():
                return cls.from_file(name)
        
        # Home directory
        home_config = Path.home() / ".cgw" / "config.yaml"
        if home_config.exists():
            return cls.from_file(home_config)
        
        # Defaults
        return cls()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        from dataclasses import asdict
        return asdict(self)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save config to YAML/JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = self.to_dict()
        
        if path.suffix in ('.yaml', '.yml'):
            try:
                import yaml
                path.write_text(yaml.dump(data, default_flow_style=False))
                return
            except ImportError:
                pass
        
        path.write_text(json.dumps(data, indent=2))


def load_config(path: Optional[str] = None) -> CGWConfig:
    """Load CGW configuration.
    
    This is the main entry point for configuration loading.
    
    Args:
        path: Optional path to config file
        
    Returns:
        CGWConfig instance with all settings
    """
    return CGWConfig.load(path)


def create_default_config() -> str:
    """Generate a default config file content as YAML string."""
    return """# CGW Coding Agent Configuration

agent:
  max_cycles: 100
  max_patches: 10
  max_test_runs: 20
  goal: "Fix failing tests"
  # repo_url: "https://github.com/user/repo.git"

llm:
  provider: deepseek
  model: deepseek-coder
  temperature: 0.2
  max_tokens: 4096
  fallback_providers:
    - openai
    - gemini

sandbox:
  image: "python:3.11-slim"
  timeout: 300
  memory_limit: "1g"

bandit:
  enabled: true
  exploration_bonus: 0.1

memory:
  enabled: true
  regression_threshold: 0.2

event_store:
  enabled: true
  retention_days: 30

dashboard:
  enabled: true
  http_port: 8765
  auto_open: true
"""
