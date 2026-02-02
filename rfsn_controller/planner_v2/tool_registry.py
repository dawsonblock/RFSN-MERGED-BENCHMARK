"""Tool Contract Registry - Catalog of allowed tools and verify recipes.

Provides a fixed catalog of:
- Allowed tool commands
- Standard verification recipes
- Tool constraints by risk level

The planner selects from this catalog - it never invents commands.
This prevents prompt injection and ensures safe tool execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ToolCategory(Enum):
    """Categories of tools."""
    
    TEST = "test"
    LINT = "lint"
    TYPE_CHECK = "type_check"
    BUILD = "build"
    FORMAT = "format"
    ANALYZE = "analyze"
    VALIDATE = "validate"


@dataclass
class ToolContract:
    """Contract defining a safe, allowed tool."""
    
    tool_id: str
    name: str
    category: ToolCategory
    command_template: str  # e.g., "pytest {target} -v"
    description: str
    
    # Constraints
    max_runtime_sec: int = 300
    allowed_exit_codes: list[int] = field(default_factory=lambda: [0])
    requires_sandbox: bool = True
    
    # Risk level this tool is appropriate for
    min_risk_level: str = "LOW"  # LOW, MED, HIGH
    
    # Output parsing
    output_parser: str | None = None  # Name of parser function
    
    def format_command(self, **kwargs) -> str:
        """Format command with provided arguments."""
        return self.command_template.format(**kwargs)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_id": self.tool_id,
            "name": self.name,
            "category": self.category.value,
            "command_template": self.command_template,
            "description": self.description,
            "max_runtime_sec": self.max_runtime_sec,
            "allowed_exit_codes": self.allowed_exit_codes,
            "requires_sandbox": self.requires_sandbox,
            "min_risk_level": self.min_risk_level,
        }


@dataclass
class VerifyRecipe:
    """Standard verification recipe for step validation."""
    
    recipe_id: str
    name: str
    tools: list[str]  # Tool IDs to run in sequence
    description: str
    
    # Recipe constraints
    fail_fast: bool = True  # Stop on first tool failure
    required_all_pass: bool = True  # All tools must pass
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "recipe_id": self.recipe_id,
            "name": self.name,
            "tools": self.tools,
            "description": self.description,
            "fail_fast": self.fail_fast,
            "required_all_pass": self.required_all_pass,
        }


class ToolContractRegistry:
    """Registry of all allowed tools and verify recipes.
    
    The planner queries this registry to select appropriate tools.
    It never invents commands - only selects from the catalog.
    """
    
    def __init__(self):
        self._tools: dict[str, ToolContract] = {}
        self._recipes: dict[str, VerifyRecipe] = {}
        self._register_default_tools()
        self._register_default_recipes()
    
    def _register_default_tools(self) -> None:
        """Register the default tool catalog."""
        
        # Python testing
        self.register_tool(ToolContract(
            tool_id="pytest",
            name="pytest",
            category=ToolCategory.TEST,
            command_template="pytest {target} -v --tb=short",
            description="Run Python tests with pytest",
            max_runtime_sec=300,
            allowed_exit_codes=[0, 1],  # 0=pass, 1=test failures
        ))
        
        self.register_tool(ToolContract(
            tool_id="pytest-focused",
            name="pytest (focused)",
            category=ToolCategory.TEST,
            command_template="pytest {target} -v --tb=short -x",
            description="Run focused test, stop on first failure",
            max_runtime_sec=60,
            allowed_exit_codes=[0, 1],
        ))
        
        # Python linting
        self.register_tool(ToolContract(
            tool_id="ruff-check",
            name="ruff check",
            category=ToolCategory.LINT,
            command_template="ruff check {target}",
            description="Fast Python linting with ruff",
            max_runtime_sec=30,
            allowed_exit_codes=[0, 1],
        ))
        
        self.register_tool(ToolContract(
            tool_id="ruff-fix",
            name="ruff check --fix",
            category=ToolCategory.LINT,
            command_template="ruff check {target} --fix",
            description="Auto-fix Python lint issues",
            max_runtime_sec=30,
            min_risk_level="MED",
        ))
        
        # Python type checking
        self.register_tool(ToolContract(
            tool_id="mypy",
            name="mypy",
            category=ToolCategory.TYPE_CHECK,
            command_template="mypy {target}",
            description="Static type checking for Python",
            max_runtime_sec=120,
            allowed_exit_codes=[0, 1],
        ))
        
        self.register_tool(ToolContract(
            tool_id="pyright",
            name="pyright",
            category=ToolCategory.TYPE_CHECK,
            command_template="pyright {target}",
            description="Fast Python type checker",
            max_runtime_sec=60,
            allowed_exit_codes=[0, 1],
        ))
        
        # Python syntax validation
        self.register_tool(ToolContract(
            tool_id="py-compile",
            name="py_compile",
            category=ToolCategory.VALIDATE,
            command_template="python -m py_compile {target}",
            description="Validate Python syntax",
            max_runtime_sec=10,
        ))
        
        # Node.js testing
        self.register_tool(ToolContract(
            tool_id="npm-test",
            name="npm test",
            category=ToolCategory.TEST,
            command_template="npm test",
            description="Run Node.js tests via npm",
            max_runtime_sec=300,
            allowed_exit_codes=[0, 1],
        ))
        
        self.register_tool(ToolContract(
            tool_id="jest",
            name="jest",
            category=ToolCategory.TEST,
            command_template="npx jest {target} --verbose",
            description="Run JavaScript tests with Jest",
            max_runtime_sec=300,
            allowed_exit_codes=[0, 1],
        ))
        
        # Node.js linting
        self.register_tool(ToolContract(
            tool_id="eslint",
            name="eslint",
            category=ToolCategory.LINT,
            command_template="npx eslint {target}",
            description="JavaScript/TypeScript linting",
            max_runtime_sec=60,
            allowed_exit_codes=[0, 1],
        ))
        
        # TypeScript
        self.register_tool(ToolContract(
            tool_id="tsc",
            name="tsc",
            category=ToolCategory.TYPE_CHECK,
            command_template="npx tsc --noEmit",
            description="TypeScript type checking",
            max_runtime_sec=120,
            allowed_exit_codes=[0, 1, 2],
        ))
        
        # Go
        self.register_tool(ToolContract(
            tool_id="go-test",
            name="go test",
            category=ToolCategory.TEST,
            command_template="go test {target} -v",
            description="Run Go tests",
            max_runtime_sec=300,
            allowed_exit_codes=[0, 1],
        ))
        
        self.register_tool(ToolContract(
            tool_id="go-vet",
            name="go vet",
            category=ToolCategory.LINT,
            command_template="go vet {target}",
            description="Go static analysis",
            max_runtime_sec=60,
        ))
        
        # Rust
        self.register_tool(ToolContract(
            tool_id="cargo-test",
            name="cargo test",
            category=ToolCategory.TEST,
            command_template="cargo test",
            description="Run Rust tests",
            max_runtime_sec=300,
            allowed_exit_codes=[0, 101],
        ))
        
        self.register_tool(ToolContract(
            tool_id="cargo-clippy",
            name="cargo clippy",
            category=ToolCategory.LINT,
            command_template="cargo clippy -- -D warnings",
            description="Rust linting with clippy",
            max_runtime_sec=120,
        ))
        
        # C/C++
        self.register_tool(ToolContract(
            tool_id="cmake-build",
            name="cmake build",
            category=ToolCategory.BUILD,
            command_template="cmake --build build",
            description="Build with CMake",
            max_runtime_sec=300,
        ))
        
        self.register_tool(ToolContract(
            tool_id="ctest",
            name="ctest",
            category=ToolCategory.TEST,
            command_template="ctest --test-dir build --output-on-failure",
            description="Run CTest",
            max_runtime_sec=300,
            allowed_exit_codes=[0, 8],
        ))
    
    def _register_default_recipes(self) -> None:
        """Register standard verification recipes."""
        
        # Python recipes
        self.register_recipe(VerifyRecipe(
            recipe_id="python-quick",
            name="Python Quick Check",
            tools=["py-compile", "ruff-check"],
            description="Fast syntax and lint check",
        ))
        
        self.register_recipe(VerifyRecipe(
            recipe_id="python-full",
            name="Python Full Check",
            tools=["py-compile", "ruff-check", "mypy", "pytest"],
            description="Complete Python validation",
        ))
        
        self.register_recipe(VerifyRecipe(
            recipe_id="python-test-only",
            name="Python Test Only",
            tools=["pytest"],
            description="Run tests only",
        ))
        
        self.register_recipe(VerifyRecipe(
            recipe_id="python-focused-test",
            name="Python Focused Test",
            tools=["pytest-focused"],
            description="Run focused test, fail fast",
        ))
        
        # Node.js recipes
        self.register_recipe(VerifyRecipe(
            recipe_id="node-quick",
            name="Node.js Quick Check",
            tools=["eslint"],
            description="Fast lint check",
        ))
        
        self.register_recipe(VerifyRecipe(
            recipe_id="node-full",
            name="Node.js Full Check",
            tools=["eslint", "tsc", "npm-test"],
            description="Complete Node.js validation",
        ))
        
        # Go recipes
        self.register_recipe(VerifyRecipe(
            recipe_id="go-full",
            name="Go Full Check",
            tools=["go-vet", "go-test"],
            description="Complete Go validation",
        ))
        
        # Rust recipes
        self.register_recipe(VerifyRecipe(
            recipe_id="rust-full",
            name="Rust Full Check",
            tools=["cargo-clippy", "cargo-test"],
            description="Complete Rust validation",
        ))
    
    def register_tool(self, tool: ToolContract) -> None:
        """Register a tool in the catalog."""
        self._tools[tool.tool_id] = tool
    
    def register_recipe(self, recipe: VerifyRecipe) -> None:
        """Register a verify recipe."""
        self._recipes[recipe.recipe_id] = recipe
    
    def get_tool(self, tool_id: str) -> ToolContract | None:
        """Get a tool by ID."""
        return self._tools.get(tool_id)
    
    def get_recipe(self, recipe_id: str) -> VerifyRecipe | None:
        """Get a recipe by ID."""
        return self._recipes.get(recipe_id)
    
    def list_tools(self, category: ToolCategory | None = None) -> list[ToolContract]:
        """List all tools, optionally filtered by category."""
        tools = list(self._tools.values())
        if category:
            tools = [t for t in tools if t.category == category]
        return tools
    
    def list_recipes(self) -> list[VerifyRecipe]:
        """List all recipes."""
        return list(self._recipes.values())
    
    def get_tools_for_risk(self, risk_level: str) -> list[ToolContract]:
        """Get tools appropriate for a risk level."""
        risk_order = {"LOW": 0, "MED": 1, "HIGH": 2}
        level = risk_order.get(risk_level, 0)
        
        return [
            t for t in self._tools.values()
            if risk_order.get(t.min_risk_level, 0) <= level
        ]
    
    def get_recipe_for_language(self, language: str, full: bool = True) -> str | None:
        """Get appropriate recipe for a language.
        
        Args:
            language: Programming language.
            full: If True, return full validation; else quick check.
            
        Returns:
            Recipe ID or None.
        """
        lang_recipes = {
            "python": ("python-full" if full else "python-quick"),
            "javascript": ("node-full" if full else "node-quick"),
            "typescript": ("node-full" if full else "node-quick"),
            "go": "go-full",
            "rust": "rust-full",
        }
        return lang_recipes.get(language.lower())
    
    def validate_command(self, command: str) -> bool:
        """Check if a command matches any registered tool.
        
        Args:
            command: Command string to validate.
            
        Returns:
            True if command matches a registered tool pattern.
        """
        command_lower = command.lower().strip()
        
        for tool in self._tools.values():
            # Extract base command from template
            base = tool.command_template.split()[0].lower()
            if command_lower.startswith(base):
                return True
        
        return False
    
    def to_dict(self) -> dict[str, Any]:
        """Export registry as dictionary."""
        return {
            "tools": {k: v.to_dict() for k, v in self._tools.items()},
            "recipes": {k: v.to_dict() for k, v in self._recipes.items()},
        }


# Singleton registry
_registry: ToolContractRegistry | None = None


def get_tool_registry() -> ToolContractRegistry:
    """Get the global tool contract registry."""
    global _registry
    if _registry is None:
        _registry = ToolContractRegistry()
    return _registry
