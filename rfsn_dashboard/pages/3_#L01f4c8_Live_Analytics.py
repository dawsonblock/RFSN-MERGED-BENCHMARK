"""RFSN Dashboard - Live Analytics Page."""

from __future__ import annotations

import json
from datetime import datetime, UTC, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

DEFAULT_OUTPUT_DIR = Path.home() / ".rfsn_upstream"


def load_historical_data(output_dir: Path, days: int = 7) -> list[dict[str, Any]]:
    """Load historical ledger data."""
    entries: list[dict[str, Any]] = []
    cutoff = datetime.now(UTC) - timedelta(days=days)

    for lf in sorted(output_dir.glob("ledger_*.jsonl"), reverse=True):
        try:
            with open(lf) as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        ts = entry.get("timestamp", "")
                        if ts:
                            try:
                                entry_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                                if entry_dt >= cutoff:
                                    entries.append(entry)
                            except ValueError:
                                entries.append(entry)
        except Exception:
            pass

    return entries


def render_live_analytics() -> None:
    """Render the live analytics page."""
    st.title("Live Analytics")

    output_dir = Path(st.session_state.get("output_dir", DEFAULT_OUTPUT_DIR))

    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Performance Metrics")
    with col2:
        days = st.selectbox("Time Range", [1, 7, 30], index=1, format_func=lambda x: f"Last {x} days")

    entries = load_historical_data(output_dir, days=days or 7)

    if not entries:
        st.warning("No data available for the selected time range.")
        return

    total = len(entries)
    successes = sum(1 for e in entries if e.get("success"))
    total_reward = sum(e.get("reward", 0) for e in entries)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Tasks", total)
    with col2:
        st.metric("Success Rate", f"{(successes / max(total, 1)) * 100:.1f}%")
    with col3:
        st.metric("Total Reward", f"{total_reward:.2f}")

    st.markdown("---")
    st.subheader("Arm Performance Leaderboard")

    arm_stats: dict[str, dict[str, Any]] = {}
    for e in entries:
        arm = e.get("arm_used", "unknown")
        if arm not in arm_stats:
            arm_stats[arm] = {"total": 0, "success": 0, "reward": 0.0}
        arm_stats[arm]["total"] += 1
        if e.get("success"):
            arm_stats[arm]["success"] += 1
        arm_stats[arm]["reward"] += e.get("reward", 0)

    if arm_stats:
        leaderboard = [
            {
                "Arm": arm,
                "Tasks": stats["total"],
                "Wins": stats["success"],
                "Win Rate %": (stats["success"] / max(stats["total"], 1)) * 100,
                "Total Reward": stats["reward"],
            }
            for arm, stats in arm_stats.items()
        ]
        df = pd.DataFrame(leaderboard).sort_values("Win Rate %", ascending=False)
        st.dataframe(df, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Live Analytics - RFSN", page_icon="📈", layout="wide")
    render_live_analytics()


if __name__ == "__main__":
    main()
