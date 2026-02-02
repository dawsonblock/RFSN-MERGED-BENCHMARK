"""RFSN Operational Dashboard - Main Application.

A comprehensive Streamlit dashboard for monitoring and controlling
the RFSN autonomous software repair system and SWE-bench learner.

Features:
- Real-time bandit arm performance visualization
- Memory index exploration
- Task ledger inspection
- System health monitoring
- Live run tracking

Usage:
    streamlit run rfsn_dashboard/app.py
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import streamlit as st

if TYPE_CHECKING:
    pass

# Page configuration
st.set_page_config(
    page_title="RFSN Dashboard",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Default paths
DEFAULT_OUTPUT_DIR = Path.home() / ".rfsn_upstream"
DEFAULT_BANDIT_DB = DEFAULT_OUTPUT_DIR / "bandit.db"
DEFAULT_MEMORY_DB = DEFAULT_OUTPUT_DIR / "memory.db"


def get_css() -> str:
    """Get custom CSS for dashboard styling."""
    return """
    <style>
        .metric-card {
            background: linear-gradient(135deg, #1e3a5f 0%, #0d1b2a 100%);
            border-radius: 12px;
            padding: 20px;
            margin: 10px 0;
            border: 1px solid rgba(99, 179, 237, 0.3);
        }
        .metric-value {
            font-size: 2.5rem;
            font-weight: bold;
            color: #63b3ed;
        }
        .metric-label {
            font-size: 0.9rem;
            color: #a0aec0;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .success-badge {
            background: linear-gradient(135deg, #2f855a 0%, #1a4731 100%);
            color: #9ae6b4;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85rem;
        }
        .failure-badge {
            background: linear-gradient(135deg, #c53030 0%, #742a2a 100%);
            color: #feb2b2;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85rem;
        }
        .sidebar-header {
            font-size: 1.5rem;
            font-weight: bold;
            margin-bottom: 1rem;
            background: linear-gradient(90deg, #63b3ed, #805ad5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
    </style>
    """


def load_bandit_stats(db_path: Path) -> list[dict[str, Any]]:
    """Load bandit arm statistics from database."""
    if not db_path.exists():
        return []

    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute("""
            SELECT arm_id, successes, failures, alpha, beta, last_updated
            FROM arms
            ORDER BY successes DESC
        """)
        rows = cur.fetchall()
        conn.close()

        return [
            {
                "arm_id": row[0],
                "successes": row[1],
                "failures": row[2],
                "alpha": row[3],
                "beta": row[4],
                "last_updated": row[5],
                "win_rate": row[1] / max(row[1] + row[2], 1) * 100,
            }
            for row in rows
        ]
    except Exception as e:
        st.error(f"Error loading bandit stats: {e}")
        return []


def load_recent_memories(db_path: Path, limit: int = 20) -> list[dict[str, Any]]:
    """Load recent memories from the memory index."""
    if not db_path.exists():
        return []

    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute("""
            SELECT id, fingerprint_hash, outcome, summary, created_at
            FROM memories
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))
        rows = cur.fetchall()
        conn.close()

        return [
            {
                "id": row[0],
                "fingerprint": row[1][:12] + "..." if row[1] and len(row[1]) > 12 else row[1],
                "outcome": row[2],
                "summary": row[3][:100] + "..." if row[3] and len(row[3]) > 100 else row[3],
                "created_at": row[4],
            }
            for row in rows
        ]
    except Exception as e:
        st.error(f"Error loading memories: {e}")
        return []


def load_ledger_entries(output_dir: Path, limit: int = 100) -> list[dict[str, Any]]:
    """Load ledger entries from all ledger files."""
    entries: list[dict[str, Any]] = []

    ledger_files = list(output_dir.glob("ledger_*.jsonl"))
    for lf in sorted(ledger_files, reverse=True):
        try:
            with open(lf) as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        entry["ledger_file"] = lf.name
                        entries.append(entry)
                        if len(entries) >= limit:
                            break
        except Exception as e:
            st.warning(f"Error reading {lf.name}: {e}")

        if len(entries) >= limit:
            break

    return entries


def render_sidebar() -> str:
    """Render the sidebar with navigation and settings."""
    with st.sidebar:
        st.markdown('<div class="sidebar-header">RFSN Dashboard</div>', unsafe_allow_html=True)
        st.markdown("---")

        page = st.radio(
            "Navigation",
            ["Overview", "Bandit Stats", "Memory Index", "Task Ledger", "Settings"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("### Quick Stats")
        output_dir = Path(st.session_state.get("output_dir", DEFAULT_OUTPUT_DIR))

        if output_dir.exists():
            ledger_count = len(list(output_dir.glob("ledger_*.jsonl")))
            st.metric("Ledger Files", ledger_count)
        else:
            st.info("No output directory found")

        return page or "Overview"


def render_overview(output_dir: Path, bandit_db: Path, memory_db: Path) -> None:
    """Render the overview page with key metrics."""
    st.title("RFSN System Overview")
    st.markdown(get_css(), unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    bandit_stats = load_bandit_stats(bandit_db)
    memories = load_recent_memories(memory_db, limit=1000)
    ledger_entries = load_ledger_entries(output_dir, limit=1000)

    total_tasks = len(ledger_entries)
    successful_tasks = sum(1 for e in ledger_entries if e.get("success"))
    total_memories = len(memories)
    active_arms = len([a for a in bandit_stats if a["successes"] + a["failures"] > 0])

    with col1:
        st.metric("Total Tasks", total_tasks)
    with col2:
        success_rate = (successful_tasks / max(total_tasks, 1)) * 100
        st.metric("Success Rate", f"{success_rate:.1f}%")
    with col3:
        st.metric("Memories", total_memories)
    with col4:
        st.metric("Active Arms", active_arms)

    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Arm Win Rates")
        if bandit_stats:
            df = pd.DataFrame(bandit_stats)
            if not df.empty:
                st.bar_chart(df.set_index("arm_id")["win_rate"])
        else:
            st.info("No bandit data available yet")

    with col_right:
        st.subheader("Recent Task Outcomes")
        if ledger_entries:
            recent = ledger_entries[:50]
            df = pd.DataFrame(recent)
            if "success" in df.columns:
                success_counts = df["success"].value_counts()
                st.bar_chart(success_counts)
        else:
            st.info("No task history available yet")


def render_bandit_stats(bandit_db: Path) -> None:
    """Render detailed bandit statistics page."""
    st.title("Bandit Arm Statistics")
    st.markdown(get_css(), unsafe_allow_html=True)

    bandit_stats = load_bandit_stats(bandit_db)

    if not bandit_stats:
        st.warning("No bandit data found. Run some tasks first!")
        st.info(f"Expected database at: `{bandit_db}`")
        return

    st.subheader("Arm Performance Comparison")
    df = pd.DataFrame(bandit_stats)
    df["total_trials"] = df["successes"] + df["failures"]
    df["win_rate_fmt"] = df["win_rate"].apply(lambda x: f"{x:.1f}%")

    st.dataframe(
        df[["arm_id", "successes", "failures", "total_trials", "win_rate_fmt", "last_updated"]],
        use_container_width=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Win Rates by Arm")
        st.bar_chart(df.set_index("arm_id")["win_rate"])
    with col2:
        st.subheader("Trial Distribution")
        st.bar_chart(df.set_index("arm_id")[["successes", "failures"]])


def render_memory_index(memory_db: Path) -> None:
    """Render the memory index exploration page."""
    st.title("Memory Index Explorer")
    st.markdown(get_css(), unsafe_allow_html=True)

    memories = load_recent_memories(memory_db, limit=100)

    if not memories:
        st.warning("No memories found.")
        return

    col1, col2 = st.columns(2)
    with col1:
        outcome_filter = st.selectbox("Filter by Outcome", ["All", "success", "failure", "rejected"])
    with col2:
        search_query = st.text_input("Search Summaries", "")

    filtered = memories
    if outcome_filter != "All":
        filtered = [m for m in filtered if m.get("outcome") == outcome_filter]
    if search_query:
        filtered = [m for m in filtered if search_query.lower() in (m.get("summary") or "").lower()]

    st.subheader(f"Memories ({len(filtered)} / {len(memories)})")
    if filtered:
        st.dataframe(pd.DataFrame(filtered), use_container_width=True)


def render_task_ledger(output_dir: Path) -> None:
    """Render the task ledger inspection page."""
    st.title("Task Ledger")
    st.markdown(get_css(), unsafe_allow_html=True)

    ledger_entries = load_ledger_entries(output_dir, limit=500)

    if not ledger_entries:
        st.warning("No ledger entries found.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        success_filter = st.selectbox("Success Filter", ["All", "Success", "Failed"])
    with col2:
        arm_options = ["All", *sorted({e.get("arm_used", "") for e in ledger_entries if e.get("arm_used")})]
        arm_filter = st.selectbox("Arm Filter", arm_options)
    with col3:
        limit = st.slider("Max Entries", 10, 500, 100)

    filtered = ledger_entries[:limit]
    if success_filter == "Success":
        filtered = [e for e in filtered if e.get("success")]
    elif success_filter == "Failed":
        filtered = [e for e in filtered if not e.get("success")]
    if arm_filter != "All":
        filtered = [e for e in filtered if e.get("arm_used") == arm_filter]

    st.subheader(f"Entries ({len(filtered)})")
    if filtered:
        st.dataframe(pd.DataFrame(filtered), use_container_width=True)


def render_settings() -> None:
    """Render the settings page."""
    st.title("Settings")
    st.markdown(get_css(), unsafe_allow_html=True)

    output_dir = st.text_input("Output Directory", value=str(st.session_state.get("output_dir", DEFAULT_OUTPUT_DIR)))
    st.session_state["output_dir"] = Path(output_dir)

    bandit_db = st.text_input("Bandit Database", value=str(st.session_state.get("bandit_db", DEFAULT_BANDIT_DB)))
    st.session_state["bandit_db"] = Path(bandit_db)

    memory_db = st.text_input("Memory Database", value=str(st.session_state.get("memory_db", DEFAULT_MEMORY_DB)))
    st.session_state["memory_db"] = Path(memory_db)

    if st.button("Refresh"):
        st.rerun()


def main() -> None:
    """Main application entry point."""
    if "output_dir" not in st.session_state:
        st.session_state["output_dir"] = DEFAULT_OUTPUT_DIR
    if "bandit_db" not in st.session_state:
        st.session_state["bandit_db"] = DEFAULT_BANDIT_DB
    if "memory_db" not in st.session_state:
        st.session_state["memory_db"] = DEFAULT_MEMORY_DB

    output_dir = Path(st.session_state["output_dir"])
    bandit_db = Path(st.session_state["bandit_db"])
    memory_db = Path(st.session_state["memory_db"])

    page = render_sidebar()

    if page == "Overview":
        render_overview(output_dir, bandit_db, memory_db)
    elif page == "Bandit Stats":
        render_bandit_stats(bandit_db)
    elif page == "Memory Index":
        render_memory_index(memory_db)
    elif page == "Task Ledger":
        render_task_ledger(output_dir)
    elif page == "Settings":
        render_settings()


if __name__ == "__main__":
    main()
