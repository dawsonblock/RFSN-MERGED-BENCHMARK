def semantic_diff_ok(diff_stats: dict, limits: dict) -> bool:
    """
    Example limits:
      max_files: 5
      max_lines: 200
      forbid_paths: ["setup.py", "ci/"]
    """

    if diff_stats["files_changed"] > limits["max_files"]:
        return False

    if diff_stats["lines_changed"] > limits["max_lines"]:
        return False

    for p in diff_stats["paths"]:
        for bad in limits.get("forbid_paths", []):
            if p.startswith(bad):
                return False

    return True
