"""Buildpack plugin system for extensible language support.

Provides a plugin architecture that allows dynamic discovery and loading
of buildpacks via Python entry points, enabling third-party extensions.
"""

from __future__ import annotations

import importlib.metadata
import logging

from .buildpacks.base import Buildpack

logger = logging.getLogger(__name__)


class BuildpackRegistry:
    """Registry for buildpack plugins.

    Discovers buildpacks via:
    1. Built-in buildpacks (rfsn_controller.buildpacks.*)
    2. Entry point plugins (entry_points(group="rfsn.buildpacks"))

    Example pyproject.toml for a plugin:
        [project.entry-points."rfsn.buildpacks"]
        scala = "my_plugin.buildpacks:ScalaBuildpack"
    """

    def __init__(self):
        """Initialize registry."""
        self._buildpacks: dict[str, type[Buildpack]] = {}
        self._load_builtin_buildpacks()
        self._load_plugin_buildpacks()

    def _load_builtin_buildpacks(self) -> None:
        """Load built-in buildpacks."""
        builtin = {
            "python": ("python_pack", "PythonBuildpack"),
            "node": ("node_pack", "NodeBuildpack"),
            "go": ("go_pack", "GoBuildpack"),
            "rust": ("rust_pack", "RustBuildpack"),
            "java": ("java_pack", "JavaBuildpack"),
            "dotnet": ("dotnet_pack", "DotnetBuildpack"),
            "cpp": ("cpp_pack", "CppBuildpack"),
            "polyrepo": ("polyrepo_pack", "PolyrepoBuildpack"),
        }

        for name, (module_name, class_name) in builtin.items():
            try:
                module = importlib.import_module(f"rfsn_controller.buildpacks.{module_name}")
                buildpack_class = getattr(module, class_name)
                self._buildpacks[name] = buildpack_class
                logger.debug(f"Loaded built-in buildpack: {name}")
            except Exception as e:
                logger.warning(f"Failed to load built-in buildpack {name}: {e}")

    def _load_plugin_buildpacks(self) -> None:
        """Load plugin buildpacks from entry points."""
        try:
            # Python 3.10+ API
            entry_points = importlib.metadata.entry_points()
            if hasattr(entry_points, "select"):
                # Python 3.10+
                buildpack_eps = entry_points.select(group="rfsn.buildpacks")
            else:
                # Python 3.9
                buildpack_eps = entry_points.get("rfsn.buildpacks", [])

            for ep in buildpack_eps:
                try:
                    buildpack_class = ep.load()
                    if not issubclass(buildpack_class, Buildpack):
                        logger.warning(f"Plugin {ep.name} does not inherit from Buildpack, skipping")
                        continue

                    self._buildpacks[ep.name] = buildpack_class
                    logger.info(f"Loaded plugin buildpack: {ep.name}")
                except Exception as e:
                    logger.warning(f"Failed to load plugin buildpack {ep.name}: {e}")
        except Exception as e:
            logger.warning(f"Failed to discover plugin buildpacks: {e}")

    def get(self, name: str) -> type[Buildpack] | None:
        """Get buildpack by name.

        Args:
            name: Buildpack name (e.g., "python", "node")

        Returns:
            Buildpack class or None if not found
        """
        return self._buildpacks.get(name)

    def list_all(self) -> list[str]:
        """List all registered buildpack names.

        Returns:
            List of buildpack names
        """
        return list(self._buildpacks.keys())

    def detect_buildpack(
        self,
        repo_dir: str,
        repo_tree: list[str],
        files: dict[str, str],
    ) -> tuple[str, Buildpack] | None:
        """Detect appropriate buildpack for a repository.

        Args:
            repo_dir: Repository directory path
            repo_tree: List of file paths in repo
            files: Mapping of filename to content

        Returns:
            Tuple of (buildpack_name, buildpack_instance) or None
        """
        best_match = None
        best_confidence = 0.0

        for name, buildpack_class in self._buildpacks.items():
            try:
                buildpack = buildpack_class()
                from .buildpacks.base import BuildpackContext

                ctx = BuildpackContext(
                    repo_dir=repo_dir,
                    repo_tree=repo_tree,
                    files=files,
                )

                result = buildpack.detect(ctx)

                if result.confidence > best_confidence:
                    best_confidence = result.confidence
                    best_match = (name, buildpack)
            except Exception as e:
                logger.debug(f"Buildpack {name} detection failed: {e}")
                continue

        if best_match and best_confidence > 0.5:
            return best_match

        return None

    def register(self, name: str, buildpack_class: type[Buildpack]) -> None:
        """Manually register a buildpack.

        Args:
            name: Buildpack name
            buildpack_class: Buildpack class
        """
        if not issubclass(buildpack_class, Buildpack):
            raise TypeError(f"{buildpack_class} must inherit from Buildpack")

        self._buildpacks[name] = buildpack_class
        logger.info(f"Registered buildpack: {name}")


# Global registry instance
_registry: BuildpackRegistry | None = None


def get_registry() -> BuildpackRegistry:
    """Get the global buildpack registry.

    Returns:
        BuildpackRegistry instance
    """
    global _registry
    if _registry is None:
        _registry = BuildpackRegistry()
    return _registry


def register_buildpack(name: str, buildpack_class: type[Buildpack]) -> None:
    """Register a custom buildpack.

    Args:
        name: Buildpack name
        buildpack_class: Buildpack class

    Example:
        from rfsn_controller.buildpacks import register_buildpack, Buildpack

        class MyBuildpack(Buildpack):
            def detect(self, ctx):
                # Implementation
                pass
            # ... other methods ...

        register_buildpack("my_language", MyBuildpack)
    """
    registry = get_registry()
    registry.register(name, buildpack_class)


def get_buildpack(name: str) -> type[Buildpack] | None:
    """Get buildpack by name.

    Args:
        name: Buildpack name

    Returns:
        Buildpack class or None
    """
    registry = get_registry()
    return registry.get(name)


def list_buildpacks() -> list[str]:
    """List all available buildpack names.

    Returns:
        List of buildpack names
    """
    registry = get_registry()
    return registry.list_all()


def detect_buildpack(
    repo_dir: str,
    repo_tree: list[str],
    files: dict[str, str],
) -> tuple[str, Buildpack] | None:
    """Auto-detect buildpack for a repository.

    Args:
        repo_dir: Repository directory path
        repo_tree: List of file paths in repo
        files: Mapping of filename to content

    Returns:
        Tuple of (buildpack_name, buildpack_instance) or None

    Example:
        match = detect_buildpack(
            repo_dir="/path/to/repo",
            repo_tree=["setup.py", "requirements.txt", "main.py"],
            files={"setup.py": "# ..."}
        )

        if match:
            name, buildpack = match
            print(f"Detected {name} buildpack")
    """
    registry = get_registry()
    return registry.detect_buildpack(repo_dir, repo_tree, files)
