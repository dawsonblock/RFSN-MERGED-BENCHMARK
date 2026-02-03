from __future__ import annotations
import ast
import os
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional

@dataclass
class ImportGraph:
    # module -> modules it imports
    edges: Dict[str, Set[str]]
    # module -> file path
    mod_to_file: Dict[str, str]
    # file path -> module
    file_to_mod: Dict[str, str]

def _is_pkg_dir(path: str) -> bool:
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "__init__.py"))

def _path_to_module(repo_root: str, py_path: str) -> Optional[str]:
    rp = os.path.relpath(py_path, repo_root).replace("\\", "/")
    if rp.startswith("../"):
        return None
    if not rp.endswith(".py"):
        return None
    rp = rp[:-3]  # strip .py
    parts = rp.split("/")
    # only accept if every parent is a package, unless top-level module
    if len(parts) > 1:
        cur = repo_root
        for d in parts[:-1]:
            cur = os.path.join(cur, d)
            if not _is_pkg_dir(cur):
                # still allow "tests/" style modules; treat as pseudo-module
                # but keep consistent naming
                break
    return ".".join(parts)

def _imports_from_ast(tree: ast.AST) -> Set[str]:
    out: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                if n.name:
                    out.add(n.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                out.add(node.module.split(".")[0])
    return out

def build_import_graph(repo_root: str, max_files: int = 6000) -> ImportGraph:
    edges: Dict[str, Set[str]] = {}
    mod_to_file: Dict[str, str] = {}
    file_to_mod: Dict[str, str] = {}

    count = 0
    for root, _, files in os.walk(repo_root):
        rnorm = root.replace("\\", "/")
        if "/.git/" in rnorm or "/.rfsn_worktrees/" in rnorm or "/.rfsn_cache/" in rnorm:
            continue
        for f in files:
            if not f.endswith(".py"):
                continue
            if f.startswith("."):
                continue
            path = os.path.join(root, f)
            mod = _path_to_module(repo_root, path)
            if not mod:
                continue

            mod_to_file[mod] = os.path.relpath(path, repo_root).replace("\\", "/")
            file_to_mod[os.path.relpath(path, repo_root).replace("\\", "/")] = mod

            count += 1
            if count > max_files:
                break
        if count > max_files:
            break

    for mod, relpath in mod_to_file.items():
        abspath = os.path.join(repo_root, relpath)
        try:
            with open(abspath, "r", encoding="utf-8") as fp:
                src = fp.read()
            tree = ast.parse(src)
            imps = _imports_from_ast(tree)
            edges[mod] = set(imps)
        except Exception:
            edges[mod] = set()

    return ImportGraph(edges=edges, mod_to_file=mod_to_file, file_to_mod=file_to_mod)

def reverse_closure(graph: ImportGraph, seeds: Set[str], depth: int = 2, cap: int = 2000) -> Set[str]:
    """
    Find modules that (transitively) import any seed module.
    Computed by scanning edges (fast enough for <= few thousand files).
    """
    if not seeds:
        return set()

    frontier = set(seeds)
    covered = set(seeds)

    for _ in range(depth):
        if len(covered) > cap:
            break
        new: Set[str] = set()
        for m, imps in graph.edges.items():
            if m in covered:
                continue
            if imps & frontier:
                new.add(m)
        if not new:
            break
        covered |= new
        frontier = new

    return covered
