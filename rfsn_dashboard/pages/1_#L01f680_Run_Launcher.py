"""RFSN Dashboard - Run Launcher Page."""

from __future__ import annotations

import subprocess
from datetime import datetime, UTC
from pathlib import Path

import streamlit as st

DEFAULT_OUTPUT_DIR = Path.home() / ".rfsn_upstream"


def render_run_launcher() -> None:
    """Render the run launcher page."""
    st.title("Run Launcher")
    
    st.markdown("Launch SWE-bench training runs directly from the dashboard.")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Run Configuration")
        tasks_file = st.text_input("Tasks File", value="tasks.jsonl")
        output_dir = st.text_input("Output Directory", value=str(DEFAULT_OUTPUT_DIR))
        max_tasks = st.number_input("Max Tasks", min_value=1, max_value=1000, value=10)

    with col2:
        st.subheader("Options")
        provider = st.selectbox("LLM Provider", ["auto", "deepseek", "gemini"])
        dry_run = st.checkbox("Dry Run", value=False)
        verbose = st.checkbox("Verbose Logging", value=True)

    st.markdown("---")

    if st.button("Launch Run", type="primary"):
        if not Path(tasks_file).exists():
            st.error(f"Tasks file not found: {tasks_file}")
        else:
            st.session_state["run_config"] = {
                "tasks_file": tasks_file,
                "output_dir": output_dir,
                "max_tasks": max_tasks,
                "provider": provider,
                "started_at": datetime.now(UTC).isoformat(),
            }

            cmd = f"python -m rfsn_upstream.swebench_runner --tasks {tasks_file} --output {output_dir} --max-tasks {max_tasks} --provider {provider}"
            if dry_run:
                cmd += " --dry-run"
            if verbose:
                cmd += " -v"

            st.success("Run command:")
            st.code(cmd, language="bash")

    if "run_config" in st.session_state:
        st.markdown("---")
        st.subheader("Current Run Config")
        st.json(st.session_state["run_config"])


def main() -> None:
    st.set_page_config(page_title="Run Launcher - RFSN", page_icon="🚀", layout="wide")
    render_run_launcher()


if __name__ == "__main__":
    main()
