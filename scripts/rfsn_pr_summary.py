\
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class Event:
    type: str
    data: Dict


def read_jsonl(path: str) -> List[Event]:
    out: List[Event] = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out.append(Event(type=obj.get("type", "event"), data=obj))
    return out


def run_git(repo_path: str, args: List[str]) -> Tuple[int, str]:
    p = subprocess.run(["git"] + args, cwd=repo_path, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout


def git_diff_stat(repo_path: str) -> str:
    rc, out = run_git(repo_path, ["diff", "--stat"])
    return out if rc == 0 else ""


def git_diff(repo_path: str, max_lines: int = 400) -> str:
    rc, out = run_git(repo_path, ["diff"])
    if rc != 0:
        return ""
    lines = out.splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines] + ["", f"... (diff truncated to {max_lines} lines)"]
    return "\n".join(lines)


def summarize(events: List[Event]) -> Dict[str, object]:
    counts: Dict[str, int] = {}
    last_status = None
    steps = 0
    failures: List[str] = []
    commands: List[str] = []

    for e in events:
        counts[e.type] = counts.get(e.type, 0) + 1

        if e.type in ("step", "iteration", "controller_step"):
            steps += 1

        if e.type in ("command", "exec", "run"):
            cmd = e.data.get("cmd") or e.data.get("argv") or e.data.get("command")
            if cmd:
                commands.append(str(cmd))

        if e.type in ("error", "failure", "verify_failed"):
            msg = e.data.get("message") or e.data.get("error") or e.data.get("reason") or "unknown failure"
            failures.append(str(msg))

        if e.type in ("done", "complete", "final"):
            last_status = e.data.get("status") or last_status

    return {
        "event_counts": counts,
        "steps": steps,
        "failures": failures[-10:],
        "commands": commands[-20:],
        "status": last_status,
    }


def write_summary(output_path: str, repo_path: str, events_path: str, goal: str) -> None:
    events = read_jsonl(events_path)
    s = summarize(events)
    stat = git_diff_stat(repo_path)
    diff = git_diff(repo_path)

    md = []
    md.append("# RFSN Run Summary\n\n")
    md.append(f"**Goal:** {goal}\n\n")
    md.append(f"**Repo:** `{repo_path}`\n\n")
    md.append(f"**Status:** `{s.get('status')}`\n\n")
    md.append(f"**Controller steps:** {s.get('steps')}\n\n")

    md.append("## Change summary\n\n")
    md.append("```text\n" + (stat.strip() or "(no diff)") + "\n```\n\n")

    md.append("## Key events\n\n")
    counts = s.get("event_counts", {})
    if isinstance(counts, dict) and counts:
        for k in sorted(counts.keys()):
            md.append(f"- `{k}`: {counts[k]}\n")
    else:
        md.append("- (no events)\n")

    fails = s.get("failures", [])
    md.append("\n## Failures (most recent)\n\n")
    if fails:
        for f in fails:
            md.append(f"- {f}\n")
    else:
        md.append("- (none recorded)\n")

    cmds = s.get("commands", [])
    md.append("\n## Commands executed (most recent)\n\n")
    if cmds:
        for c in cmds:
            md.append(f"- `{c}`\n")
    else:
        md.append("- (none recorded)\n")

    md.append("\n## Diff (truncated)\n\n")
    md.append("```diff\n" + (diff.strip() or "(no diff)") + "\n```\n")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("".join(md))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--goal", required=True)
    ap.add_argument("--out", default="artifacts/RFSN_SUMMARY.md")
    args = ap.parse_args()
    write_summary(args.out, args.repo, args.events, args.goal)
    print(args.out)
