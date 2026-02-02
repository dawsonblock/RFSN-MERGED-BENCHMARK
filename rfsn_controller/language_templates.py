"""Command templates for different languages.
from __future__ import annotations

Provides safe, templated commands for install, test, and build operations
across supported languages.
"""

from dataclasses import dataclass
from enum import Enum


class Language(Enum):
    """Supported languages."""

    PYTHON = "python"
    NODE = "node"
    GO = "go"
    RUST = "rust"
    JAVA = "java"
    DOTNET = "dotnet"


@dataclass
class CommandTemplates:
    """Command templates for a language."""

    install: str
    test: str
    build: str | None
    lint: str | None
    typecheck: str | None


# Language-specific command templates
TEMPLATES: dict[Language, CommandTemplates] = {
    Language.PYTHON: CommandTemplates(
        install="python -m pip install -e .",
        test="python -m pytest -q",
        build=None,
        lint="ruff check .",
        typecheck="mypy .",
    ),
    Language.NODE: CommandTemplates(
        install="npm ci",
        test="npm test",
        build="npm run build",
        lint="npm run lint",
        typecheck="npm run typecheck",
    ),
    Language.GO: CommandTemplates(
        install="go mod download",
        test="go test ./...",
        build="go build ./...",
        lint="golangci-lint run",
        typecheck=None,
    ),
    Language.RUST: CommandTemplates(
        install="cargo fetch",
        test="cargo test",
        build="cargo build --release",
        lint="cargo clippy",
        typecheck=None,
    ),
    Language.JAVA: CommandTemplates(
        install="mvn dependency:resolve",
        test="mvn test",
        build="mvn package",
        lint="mvn checkstyle:check",
        typecheck="mvn compile",
    ),
    Language.DOTNET: CommandTemplates(
        install="dotnet restore",
        test="dotnet test",
        build="dotnet build --configuration Release",
        lint=None,
        typecheck=None,
    ),
}


# Buildpack Docker images for each language
BUILDPACK_IMAGES: dict[Language, str] = {
    Language.PYTHON: "python:3.11-slim",
    Language.NODE: "node:20-slim",
    Language.GO: "golang:1.22-slim",
    Language.RUST: "rust:1.78-slim",
    Language.JAVA: "eclipse-temurin:17-jdk",
    Language.DOTNET: "mcr.microsoft.com/dotnet/sdk:8.0",
}


def get_templates(language: Language) -> CommandTemplates:
    """Get command templates for a language.

    Args:
        language: The language to get templates for.

    Returns:
        CommandTemplates for the language.
    """
    return TEMPLATES.get(language, TEMPLATES[Language.PYTHON])


def get_buildpack_image(language: Language) -> str:
    """Get buildpack Docker image for a language.

    Args:
        language: The language to get image for.

    Returns:
        Docker image name.
    """
    return BUILDPACK_IMAGES.get(language, BUILDPACK_IMAGES[Language.PYTHON])


def get_all_supported_languages() -> list[Language]:
    """Get list of all supported languages.

    Returns:
        List of Language enums.
    """
    return list(Language)
