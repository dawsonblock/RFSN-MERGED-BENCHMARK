"""Workspace resolver for monorepo support.
from __future__ import annotations

This module provides functionality to resolve workspaces in monorepos,
handling tools like npm workspaces, poetry workspaces, and Go modules.
"""

import os
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class WorkspaceInfo:
    """Information about a workspace in a monorepo."""

    name: str
    path: str
    language: str
    has_tests: bool
    test_command: str | None = None
    dependencies: list[str] | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.metadata is None:
            self.metadata = {}


class WorkspaceResolver:
    """Resolves workspaces in monorepos."""

    def __init__(self, repo_dir: str):
        """Initialize the workspace resolver.

        Args:
            repo_dir: Path to the repository root.
        """
        self.repo_dir = repo_dir

    def resolve(self) -> list[WorkspaceInfo]:
        """Resolve all workspaces in the repository.

        Returns:
            List of WorkspaceInfo objects.
        """
        # Try different workspace resolution strategies
        workspaces: list[WorkspaceInfo] = []

        # Try npm/yarn/pnpm workspaces
        npm_workspaces = self._resolve_npm_workspaces()
        if npm_workspaces:
            workspaces.extend(npm_workspaces)

        # Try poetry workspaces
        poetry_workspaces = self._resolve_poetry_workspaces()
        if poetry_workspaces:
            workspaces.extend(poetry_workspaces)

        # Try Go modules
        go_modules = self._resolve_go_modules()
        if go_modules:
            workspaces.extend(go_modules)

        # Try Rust workspaces
        rust_workspaces = self._resolve_rust_workspaces()
        if rust_workspaces:
            workspaces.extend(rust_workspaces)

        # Try Maven/Gradle multi-module projects
        java_modules = self._resolve_java_modules()
        if java_modules:
            workspaces.extend(java_modules)

        # Try .NET solutions
        dotnet_projects = self._resolve_dotnet_projects()
        if dotnet_projects:
            workspaces.extend(dotnet_projects)

        return workspaces

    def _resolve_npm_workspaces(self) -> list[WorkspaceInfo]:
        """Resolve npm/yarn/pnpm workspaces.

        Returns:
            List of WorkspaceInfo objects for Node.js workspaces.
        """
        workspaces: list[WorkspaceInfo] = []

        # Check for package.json with workspaces field
        package_json_path = os.path.join(self.repo_dir, "package.json")
        if not os.path.exists(package_json_path):
            return workspaces

        try:
            import json

            with open(package_json_path) as f:
                package_json = json.load(f)

            workspaces_field = package_json.get("workspaces")
            if not workspaces_field:
                return workspaces

            # Normalize workspaces field (can be array or object with packages)
            if isinstance(workspaces_field, dict):
                workspace_patterns = workspaces_field.get("packages", [])
            else:
                workspace_patterns = workspaces_field

            # Find all directories matching workspace patterns
            for pattern in workspace_patterns:
                # Convert glob pattern to regex
                regex_pattern = pattern.replace("*", ".*")
                for root, dirs, _files in os.walk(self.repo_dir):
                    # Skip node_modules and hidden dirs
                    dirs[:] = [d for d in dirs if d != "node_modules" and not d.startswith(".")]

                    rel_path = os.path.relpath(root, self.repo_dir)
                    if re.match(regex_pattern, rel_path):
                        # Check if this directory has a package.json
                        ws_package_json = os.path.join(root, "package.json")
                        if os.path.exists(ws_package_json):
                            with open(ws_package_json) as f:
                                ws_package = json.load(f)

                            # Check for test scripts
                            test_script = ws_package.get("scripts", {}).get("test")
                            has_tests = test_script is not None

                            workspaces.append(
                                WorkspaceInfo(
                                    name=ws_package.get("name", rel_path),
                                    path=rel_path,
                                    language="node",
                                    has_tests=has_tests,
                                    test_command=f"cd {rel_path} && npm test"
                                    if has_tests
                                    else None,
                                    dependencies=list(ws_package.get("dependencies", {}).keys()),
                                    metadata={
                                        "package_manager": self._detect_npm_package_manager(root)
                                    },
                                )
                            )
        except Exception:
            pass

        return workspaces

    def _resolve_poetry_workspaces(self) -> list[WorkspaceInfo]:
        """Resolve poetry workspaces.

        Returns:
            List of WorkspaceInfo objects for Python workspaces.
        """
        workspaces: list[WorkspaceInfo] = []

        # Check for pyproject.toml with poetry.plugins
        pyproject_path = os.path.join(self.repo_dir, "pyproject.toml")
        if not os.path.exists(pyproject_path):
            return workspaces

        try:
            import toml  # type: ignore

            with open(pyproject_path) as f:
                pyproject = toml.load(f)

            poetry_section = pyproject.get("tool", {}).get("poetry", {})
            if not poetry_section:
                return workspaces

            # Check for workspaces (poetry >= 1.2)
            workspaces_field = poetry_section.get("workspaces")
            if not workspaces_field:
                return workspaces

            # Find all pyproject.toml files in workspace directories
            for pattern in workspaces_field:
                regex_pattern = pattern.replace("*", ".*")
                for root, dirs, _files in os.walk(self.repo_dir):
                    dirs[:] = [d for d in dirs if d != ".venv" and not d.startswith(".")]

                    rel_path = os.path.relpath(root, self.repo_dir)
                    if re.match(regex_pattern, rel_path):
                        ws_pyproject = os.path.join(root, "pyproject.toml")
                        if os.path.exists(ws_pyproject) and ws_pyproject != pyproject_path:
                            with open(ws_pyproject) as f:
                                ws_config = toml.load(f)

                            ws_poetry = ws_config.get("tool", {}).get("poetry", {})
                            has_tests = ws_poetry.get("group", {}).get("test") is not None

                            workspaces.append(
                                WorkspaceInfo(
                                    name=ws_poetry.get("name", rel_path),
                                    path=rel_path,
                                    language="python",
                                    has_tests=has_tests,
                                    test_command=f"cd {rel_path} && poetry test"
                                    if has_tests
                                    else None,
                                    dependencies=list(ws_poetry.get("dependencies", {}).keys()),
                                    metadata={"poetry": True},
                                )
                            )
        except Exception:
            pass

        return workspaces

    def _resolve_go_modules(self) -> list[WorkspaceInfo]:
        """Resolve Go modules.

        Returns:
            List of WorkspaceInfo objects for Go modules.
        """
        workspaces: list[WorkspaceInfo] = []

        # Check for go.mod
        go_mod_path = os.path.join(self.repo_dir, "go.mod")
        if not os.path.exists(go_mod_path):
            return workspaces

        try:
            with open(go_mod_path) as f:
                content = f.read()

            # Parse module name
            module_match = re.search(r"^module\s+([^\s]+)", content, re.MULTILINE)
            if not module_match:
                return workspaces

            module_name = module_match.group(1)

            # Check for test files
            has_tests = False
            for _root, _dirs, files in os.walk(self.repo_dir):
                if any(f.endswith("_test.go") for f in files):
                    has_tests = True
                    break

            workspaces.append(
                WorkspaceInfo(
                    name=module_name,
                    path=".",
                    language="go",
                    has_tests=has_tests,
                    test_command="go test ./..." if has_tests else None,
                    dependencies=self._parse_go_dependencies(content),
                    metadata={"module": module_name},
                )
            )
        except Exception:
            pass

        return workspaces

    def _resolve_rust_workspaces(self) -> list[WorkspaceInfo]:
        """Resolve Rust workspaces.

        Returns:
            List of WorkspaceInfo objects for Rust workspaces.
        """
        workspaces: list[WorkspaceInfo] = []

        # Check for Cargo.toml with workspace section
        cargo_path = os.path.join(self.repo_dir, "Cargo.toml")
        if not os.path.exists(cargo_path):
            return workspaces

        try:
            import toml

            with open(cargo_path) as f:
                cargo_config = toml.load(f)

            workspace_section = cargo_config.get("workspace")
            if not workspace_section:
                return workspaces

            # Get workspace members
            members = workspace_section.get("members", [])
            if not members:
                return workspaces

            # Resolve each workspace member
            for member in members:
                member_path = os.path.join(self.repo_dir, member)
                member_cargo = os.path.join(member_path, "Cargo.toml")

                if os.path.exists(member_cargo):
                    with open(member_cargo) as f:
                        member_config = toml.load(f)

                    package = member_config.get("package", {})
                    name = package.get("name", member)
                    has_tests = package.get("test", True)

                    workspaces.append(
                        WorkspaceInfo(
                            name=name,
                            path=member,
                            language="rust",
                            has_tests=has_tests,
                            test_command=f"cd {member} && cargo test" if has_tests else None,
                            dependencies=list(member_config.get("dependencies", {}).keys()),
                            metadata={"crate": name},
                        )
                    )
        except Exception:
            pass

        return workspaces

    def _resolve_java_modules(self) -> list[WorkspaceInfo]:
        """Resolve Maven/Gradle multi-module projects.

        Returns:
            List of WorkspaceInfo objects for Java modules.
        """
        workspaces = []

        # Check for Maven multi-module project
        pom_path = os.path.join(self.repo_dir, "pom.xml")
        if os.path.exists(pom_path):
            workspaces.extend(self._resolve_maven_modules())

        # Check for Gradle multi-module project
        build_gradle = os.path.join(self.repo_dir, "build.gradle")
        if os.path.exists(build_gradle):
            workspaces.extend(self._resolve_gradle_modules())

        return workspaces

    def _resolve_maven_modules(self) -> list[WorkspaceInfo]:
        """Resolve Maven modules.

        Returns:
            List of WorkspaceInfo objects for Maven modules.
        """
        workspaces: list[WorkspaceInfo] = []

        try:
            import xml.etree.ElementTree as ET

            tree = ET.parse(os.path.join(self.repo_dir, "pom.xml"))
            root = tree.getroot()

            # Check for modules
            modules = root.find("{http://maven.apache.org/POM/4.0.0}modules")
            if modules is None:
                return workspaces

            for module in modules.findall("{http://maven.apache.org/POM/4.0.0}module"):
                module_path = module.text
                if not module_path:
                    continue

                module_pom = os.path.join(self.repo_dir, module_path, "pom.xml")

                if os.path.exists(module_pom):
                    module_tree = ET.parse(module_pom)
                    module_root = module_tree.getroot()

                    artifact_id = module_root.find("{http://maven.apache.org/POM/4.0.0}artifactId")
                    name = artifact_id.text if artifact_id is not None else module_path

                    # Check for test directory
                    test_dir = os.path.join(self.repo_dir, module_path, "src/test")
                    has_tests = os.path.exists(test_dir)

                    workspaces.append(
                        WorkspaceInfo(
                            name=name or module_path,
                            path=module_path,
                            language="java",
                            has_tests=has_tests,
                            test_command=f"cd {module_path} && mvn test" if has_tests else None,
                            metadata={"build_tool": "maven"},
                        )
                    )
        except Exception:
            pass

        return workspaces

    def _resolve_gradle_modules(self) -> list[WorkspaceInfo]:
        """Resolve Gradle modules.

        Returns:
            List of WorkspaceInfo objects for Gradle modules.
        """
        workspaces: list[WorkspaceInfo] = []

        try:
            # Parse settings.gradle or settings.gradle.kts
            settings_files = ["settings.gradle", "settings.gradle.kts"]
            for settings_file in settings_files:
                settings_path = os.path.join(self.repo_dir, settings_file)
                if os.path.exists(settings_path):
                    with open(settings_path) as f:
                        content = f.read()

                    # Find include statements
                    includes = re.findall(r'include\s*[\'"]([^\'"]+)[\'"]', content)
                    for include in includes:
                        module_path = include.replace(":", "/")
                        module_build = os.path.join(self.repo_dir, module_path, "build.gradle")

                        if os.path.exists(module_build):
                            # Check for test directory
                            test_dir = os.path.join(self.repo_dir, module_path, "src/test")
                            has_tests = os.path.exists(test_dir)

                            workspaces.append(
                                WorkspaceInfo(
                                    name=str(include),
                                    path=str(module_path),
                                    language="java",
                                    has_tests=has_tests,
                                    test_command=f"cd {module_path} && gradle test"
                                    if has_tests
                                    else None,
                                    metadata={"build_tool": "gradle"},
                                )
                            )
                    break
        except Exception:
            pass

        return workspaces

    def _resolve_dotnet_projects(self) -> list[WorkspaceInfo]:
        """Resolve .NET solution projects.

        Returns:
            List of WorkspaceInfo objects for .NET projects.
        """
        workspaces: list[WorkspaceInfo] = []

        # Find all .sln files
        for root, _dirs, files in os.walk(self.repo_dir):
            for file in files:
                if file.endswith(".sln"):
                    sln_path = os.path.join(root, file)
                    workspaces.extend(self._parse_solution_file(sln_path))

        return workspaces

    def _parse_solution_file(self, sln_path: str) -> list[WorkspaceInfo]:
        """Parse a .NET solution file.

        Args:
            sln_path: Path to the .sln file.

        Returns:
            List of WorkspaceInfo objects for projects in the solution.
        """
        workspaces: list[WorkspaceInfo] = []

        try:
            with open(sln_path) as f:
                content = f.read()

            # Find project references
            project_pattern = r'Project\("{[^"]+}"\)\s*=\s*"([^"]+)",\s*"([^"]+)"'
            for match in re.finditer(project_pattern, content):
                name = match.group(1)
                project_path = match.group(2)

                # Convert relative path to absolute
                if not os.path.isabs(project_path):
                    sln_dir = os.path.dirname(sln_path)
                    project_path = os.path.join(sln_dir, project_path)

                # Normalize path relative to repo
                rel_path = os.path.relpath(project_path, self.repo_dir)
                proj_dir = os.path.dirname(rel_path)

                # Check for test project
                is_test = "test" in name.lower() or "tests" in name.lower()

                workspaces.append(
                    WorkspaceInfo(
                        name=name,
                        path=proj_dir,
                        language="dotnet",
                        has_tests=is_test,
                        test_command=f"cd {proj_dir} && dotnet test" if is_test else None,
                        metadata={"project_type": "test" if is_test else "app"},
                    )
                )
        except Exception:
            pass

        return workspaces

    def _detect_npm_package_manager(self, dir_path: str) -> str:
        """Detect which npm package manager is being used.

        Args:
            dir_path: Directory to check.

        Returns:
            Package manager name (npm, yarn, pnpm, bun).
        """
        # Check for lock files
        lock_files = {
            "yarn.lock": "yarn",
            "package-lock.json": "npm",
            "pnpm-lock.yaml": "pnpm",
            "bun.lockb": "bun",
        }

        for lock_file, pm in lock_files.items():
            if os.path.exists(os.path.join(dir_path, lock_file)):
                return pm

        return "npm"  # Default

    def _parse_go_dependencies(self, go_mod_content: str) -> list[str]:
        """Parse dependencies from go.mod content.

        Args:
            go_mod_content: Content of go.mod file.

        Returns:
            List of dependency module paths.
        """
        dependencies = []

        # Find require blocks
        require_pattern = r"require\s*\((.*?)\)"
        for match in re.finditer(require_pattern, go_mod_content, re.DOTALL):
            block = match.group(1)
            for line in block.split("\n"):
                line = line.strip()
                if line and not line.startswith("//"):
                    parts = line.split()
                    if parts and not parts[0].startswith("go"):
                        dependencies.append(parts[0])

        return dependencies


def resolve_workspaces(repo_dir: str) -> list[WorkspaceInfo]:
    """Resolve workspaces in a repository.

    Args:
        repo_dir: Path to the repository root.

    Returns:
        List of WorkspaceInfo objects.
    """
    resolver = WorkspaceResolver(repo_dir)
    return resolver.resolve()
