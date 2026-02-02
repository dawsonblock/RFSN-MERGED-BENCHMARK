"""RFSN Dashboard - System Health Page."""

from __future__ import annotations

import os
import platform
import sqlite3
from pathlib import Path
from typing import Any

import streamlit as st

DEFAULT_OUTPUT_DIR = Path.home() / ".rfsn_upstream"
DEFAULT_BANDIT_DB = DEFAULT_OUTPUT_DIR / "bandit.db"
DEFAULT_MEMORY_DB = DEFAULT_OUTPUT_DIR / "memory.db"


def check_database_health(db_path: Path) -> dict[str, Any]:
    """Check health of a SQLite database."""
    if not db_path.exists():
        return {"status": "missing", "message": "Database not found"}

    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute("PRAGMA integrity_check")
        integrity = cur.fetchone()[0]
        size_bytes = db_path.stat().st_size
        cur.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
        table_count = cur.fetchone()[0]
        conn.close()

        return {
            "status": "healthy" if integrity == "ok" else "corrupted",
            "integrity": integrity,
            "size_mb": size_bytes / (1024 * 1024),
            "table_count": table_count,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def render_system_health() -> None:
    """Render the system health page."""
    st.title("System Health")

    output_dir = Path(st.session_state.get("output_dir", DEFAULT_OUTPUT_DIR))
    bandit_db = Path(st.session_state.get("bandit_db", DEFAULT_BANDIT_DB))
    memory_db = Path(st.session_state.get("memory_db", DEFAULT_MEMORY_DB))

    bandit_health = check_database_health(bandit_db)
    memory_health = check_database_health(memory_db)

    st.subheader("Component Health")
    col1, col2, col3 = st.columns(3)

    with col1:
        if output_dir.exists():
            st.success("Output Directory OK")
        else:
            st.error("Output Directory Missing")

    with col2:
        if bandit_health["status"] == "healthy":
            st.success("Bandit Database OK")
        else:
            st.warning(f"Bandit: {bandit_health['status']}")

    with col3:
        if memory_health["status"] == "healthy":
            st.success("Memory Database OK")
        else:
            st.warning(f"Memory: {memory_health['status']}")

    st.markdown("---")
    st.subheader("API Key Status")
    col1, col2, col3 = st.columns(3)

    with col1:
        if os.getenv("DEEPSEEK_API_KEY"):
            st.success("DEEPSEEK_API_KEY Set")
        else:
            st.error("DEEPSEEK_API_KEY Missing")

    with col2:
        if os.getenv("GEMINI_API_KEY"):
            st.success("GEMINI_API_KEY Set")
        else:
            st.error("GEMINI_API_KEY Missing")

    with col3:
        if os.getenv("GOOGLE_API_KEY"):
            st.success("GOOGLE_API_KEY Set")
        else:
            st.info("GOOGLE_API_KEY (optional)")

    st.markdown("---")
    st.subheader("System Information")
    st.metric("Platform", platform.system())
    st.metric("Python Version", platform.python_version())


def main() -> None:
    st.set_page_config(page_title="System Health - RFSN", page_icon="🏥", layout="wide")
    render_system_health()


if __name__ == "__main__":
    main()
