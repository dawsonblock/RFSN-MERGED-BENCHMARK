"""Advisory cognition modules for the coding agent.

These modules are **advisory only**: they never execute tools, never run shell,
and never apply patches. Their only allowed effect is to adjust candidate
scores (saliency/urgency/surprise) before the thalamic gate makes its
single-winner selection.
"""
