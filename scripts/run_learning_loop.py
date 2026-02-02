from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List

from learning.swebench_learner import SWEBenchLearner, classify_bucket
from retrieval.failure_index import FailureIndex
from learning.outcomes import Outcome

from eval.swebench import load_tasks
from orchestrator.episode_runner import run_one_task
from agent.llm_patcher import get_llm_patch_fn


def write_report(path: str, summary: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    md_path = os.path.splitext(path)[0] + ".md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# RFSN Learning Summary\n\n")
        f.write("## Bucket rank\n\n")
        for row in summary.get("bucket_rank", []):
            f.write(f"- {row['bucket']}: mean={row['mean_reward']:.3f}, success={row['success']}/{row['n']}\n")
        f.write("\n## Template rank\n\n")
        for row in summary.get("template_rank", []):
            f.write(f"- {row['template']}: mean={row['mean_reward']:.3f}, success={row['success']}/{row['n']}\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="swebench_lite", choices=["swebench_lite", "swebench_verified", "swebench_full"])
    ap.add_argument("--max_tasks", type=int, default=25)
    ap.add_argument("--model", default="deepseek", help="Model to use for patching")
    ap.add_argument("--workers", type=int, default=1)  # Serial by default
    ap.add_argument("--state_dir", default=".rfsn_state")
    args = ap.parse_args()

    learner = SWEBenchLearner(state_dir=args.state_dir)
    # failure_index is used by run_one_task internally, but we can verify it exists
    os.makedirs(args.state_dir, exist_ok=True)

    print(f"Loading tasks from {args.dataset}...")
    tasks = load_tasks(args.dataset, max_tasks=args.max_tasks)
    
    llm_patch_fn = get_llm_patch_fn(args.model)

    solved = 0
    attempted = 0

    for t in tasks:
        attempted += 1
        
        # Extract task details (handling SWEBenchTask object or dict)
        task_dict = t.to_dict() if hasattr(t, "to_dict") else t
        
        task_id = str(task_dict.get("task_id") or task_dict.get("instance_id") or f"task_{attempted}")
        repo = str(task_dict.get("repo") or task_dict.get("repo_name") or "unknown")
        test_output_start = str(task_dict.get("problem_statement") or "") # We classify based on problem statement or logs? 
        # User said: "Classify failure into a SWE-bench taxonomy bucket (test failure...)" from test output text.
        # But initially we only have problem statement + test patch.
        # We run baseline tests inside run_one_task.
        # But we need to classify BEFORE proposing?
        # classify_bucket takes `test_output`. 
        # In run_one_task, baseline is run early.
        # Ideally, we should run baseline first, then classify?
        # But learner chooses strategy BEFORE loop.
        # We can pass empty string or problem statement?
        # User code: `test_output = str(t.get("test_output") or t.get("fail_log") or "")`
        # SWE-bench tasks often come with fail logs. If not, we might be flying blind initially.
        # I'll use problem statement for now if fail_log is missing.
        initial_log = str(task_dict.get("fail_log") or "")
        bucket = classify_bucket(initial_log)

        # Pick strategy/template (upstream guidance)
        strategy = learner.choose_strategy()
        template = learner.strategy_to_template(strategy)

        # Pick planner
        planner_options = ["planner_v1"] 
        planner_name = learner.choose_planner(planner_options)

        # Inject upstream hints
        task_dict["_upstream"] = {
            "bucket": bucket,
            "strategy": strategy,
            "template": template,
            "planner": planner_name,
        }
        
        print(f"[{attempted}/{len(tasks)}] Task {task_id}: {bucket} -> {strategy} ({planner_name})")

        # Run episode
        # Construct repo URL
        repo_url = f"https://github.com/{repo}.git" if "github.com" not in repo else repo
        
        # Define callback to record each attempt
        def record_attempt(res):
            outcome = Outcome(
                passed=res.passed,
                test_delta=getattr(res, "test_delta", 0),
                runtime=getattr(res, "runtime", 0.0),
                error_message=res.reason or ""
            )
            learner.record_episode(
                task_id=task_id,
                repo=repo,
                bucket=bucket,
                planner=planner_name,
                strategy=strategy,
                template=template,
                outcome=outcome,
                patch_size=getattr(res, "patch_size", 0),
                files_touched=getattr(res, "files_touched", 0),
                extra={"dataset": args.dataset, "gate_rejections": res.gate_rejections},
            )

        run_res = run_one_task(
            task=task_dict,
            repo_url=repo_url,
            llm_patch_fn=llm_patch_fn,
            max_attempts=6,
            record_callback=record_attempt
        )
        
        status = "SOLVED" if run_res.passed else "FAILED"
        print(f"  Result: {status}")

        if run_res.passed:
            solved += 1

    summary = learner.summarize()
    summary["attempted"] = attempted
    summary["solved"] = solved
    summary["solve_rate"] = (solved / attempted) if attempted else 0.0

    report_path = os.path.join(args.state_dir, "reports", "learning_summary.json")
    write_report(report_path, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
