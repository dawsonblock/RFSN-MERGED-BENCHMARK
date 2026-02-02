# SWE-bench datasets

This repo runs in **STRICT benchmark mode** by default.

Place ONE of these files here:

- `datasets/swebench_lite.jsonl`
- `datasets/swebench_verified.jsonl`
- `datasets/swebench_full.jsonl`

If the requested dataset is missing, evaluation **fails hard**.
No sample fallback is allowed in CI.

## Acquiring Datasets

SWE-bench datasets are available from the official repository:
<https://github.com/princeton-nlp/SWE-bench>

To copy a dataset into this directory:

```bash
python scripts/fetch_swebench.py /path/to/swebench_lite.jsonl
```

## Dataset Format

Each line is a JSON object with:

- `instance_id`: Unique task identifier
- `repo`: GitHub repo (e.g., "django/django")
- `base_commit`: Commit SHA to checkout
- `problem_statement`: Description of the bug
- `test_patch`: Patch that adds failing tests
- `hints` (optional): Metadata including test commands
