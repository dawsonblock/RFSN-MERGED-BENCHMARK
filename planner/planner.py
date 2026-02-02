"""Core planner implementation."""
from __future__ import annotations
from .spec import Plan, RepairStep
from typing import Dict, Any, List
import uuid


def generate_plan(task: Dict[str, Any], retrieved_memory: Dict[str, Any]) -> Plan:
    """
    Deterministic plan generator wrapper.
    
    This upstream planner:
    1. Analyzes failure type (via hypotheses in metadata/task)
    2. Incorporates retrieved patterns
    3. Outputs a structural plan for the LLM to follow
    """
    failing_files = task.get("failing_files", [])
    if not failing_files:
        failing_files = task.get("files", ["unknown.py"])
        
    task_desc = task.get("description", task.get("problem_statement", "unknown bug"))
    
    steps: List[RepairStep] = []
    
    # 1. Standard identification step
    steps.append(RepairStep(
        intent="Analyze failure and locate bug",
        files=failing_files,
        hypothesis="Bug is likely in the failing files or their dependencies"
    ))

    # 2. Add retrieval-informed steps
    retrieval_hits = retrieved_memory.get("similar_failures", [])
    if retrieval_hits:
        best_hit = retrieval_hits[0]
        steps.append(RepairStep(
            intent=f"Apply similar fix pattern: {best_hit.get('patch_summary', 'unknown')[:50]}...",
            files=failing_files,
            hypothesis=f"Failure matches known pattern (score: {best_hit.get('score', 0):.2f})"
        ))
    
    # 3. Add hypothesis-driven steps (if available in task metadata)
    hypotheses = task.get("hypotheses", [])
    if hypotheses:
        # If we have specific hypotheses (e.g. from Classifier), use the top one
        top_h = hypotheses[0]
        steps.append(RepairStep(
            intent=f"Address {top_h.kind} error",
            files=failing_files,
            hypothesis=top_h.reasoning
        ))
    else:
        # Generic fix step
        steps.append(RepairStep(
            intent="Generate minimal fix",
            files=failing_files,
            hypothesis="Code logic needs correction"
        ))

    return Plan(
        task_id=str(uuid.uuid4()),
        bug_summary=task_desc,
        steps=steps,
        confidence=0.5 if retrieval_hits else 0.3,
        metadata={
            "source": "planner_v2", 
            "retrieval_count": len(retrieval_hits),
            "hypotheses_count": len(hypotheses)
        }
    )
