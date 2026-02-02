"""Buildpacks for multi-language repository support.
from __future__ import annotations

This module provides buildpack implementations for various programming
languages, enabling the RFSN Sandbox Controller to work with any public
repository regardless of language.
"""

from .base import (
    Buildpack,
    BuildpackContext,
    BuildpackType,
    DetectResult,
    FailureInfo,
    Step,
    TestPlan,
)
from .cpp_pack import CppBuildpack
from .dotnet_pack import DotnetBuildpack
from .go_pack import GoBuildpack
from .java_pack import JavaBuildpack
from .node_pack import NodeBuildpack
from .polyrepo_pack import PolyrepoBuildpack
from .python_pack import PythonBuildpack
from .rust_pack import RustBuildpack

__all__ = [
    "Buildpack",
    "BuildpackType",
    "BuildpackContext",
    "DetectResult",
    "Step",
    "TestPlan",
    "FailureInfo",
    "PythonBuildpack",
    "NodeBuildpack",
    "GoBuildpack",
    "RustBuildpack",
    "JavaBuildpack",
    "DotnetBuildpack",
    "CppBuildpack",
    "PolyrepoBuildpack",
    "get_buildpack",
    "get_all_buildpacks",
]


def get_buildpack(buildpack_type: BuildpackType) -> Buildpack:
    """Get a buildpack instance by type.

    Args:
        buildpack_type: The type of buildpack to get.

    Returns:
        A buildpack instance.

    Raises:
        ValueError: If the buildpack type is unknown.
    """
    buildpacks = {
        BuildpackType.PYTHON: PythonBuildpack,
        BuildpackType.NODE: NodeBuildpack,
        BuildpackType.GO: GoBuildpack,
        BuildpackType.RUST: RustBuildpack,
        BuildpackType.JAVA: JavaBuildpack,
        BuildpackType.DOTNET: DotnetBuildpack,
        BuildpackType.CPP: CppBuildpack,
        BuildpackType.POLYREPO: PolyrepoBuildpack,
    }

    buildpack_class = buildpacks.get(buildpack_type)
    if buildpack_class is None:
        raise ValueError(f"Unknown buildpack type: {buildpack_type}")

    return buildpack_class()


def get_all_buildpacks() -> list[Buildpack]:
    """Get all available buildpack instances ordered by detection priority.

    Returns:
        List of all buildpack instances in priority order.
    """
    # Order by commonality for early termination optimization
    return [
        PythonBuildpack(),  # Most common
        NodeBuildpack(),  # Second most common
        JavaBuildpack(),  # Common in enterprise
        GoBuildpack(),  # Growing popularity
        CppBuildpack(),  # Systems programming
        RustBuildpack(),  # Less common
        DotnetBuildpack(),  # Enterprise
        PolyrepoBuildpack(),  # Last, expensive to detect
    ]
