\
# Artifacts publishing and signed runs

RFSN emits an `artifacts/` directory (events, summaries, logs, etc). This repo includes two utilities:

- `scripts/rfsn_publish_artifacts.py`  
  Creates a `manifest.json` (sha256 per artifact), optionally signs it, then publishes artifacts.
- `scripts/rfsn_verify_signature.py`  
  Verifies the signature and checks artifact hashes match the manifest.

## Signing model

This implementation uses **HMAC-SHA256** over `manifest.json`.

Set the signing key in an env var:

- `RFSN_SIGNING_KEY` (recommended)
- or change the env name with `RFSN_SIGNING_KEY_ENV`

### Create + sign + publish locally

```bash
export RFSN_SIGNING_KEY="change-me"
python scripts/rfsn_publish_artifacts.py --artifacts artifacts --publish-backend local --publish-local-dir artifacts/published
```

### Verify

```bash
export RFSN_SIGNING_KEY="change-me"
python scripts/rfsn_verify_signature.py --artifacts artifacts
```

## Publish to S3

Install boto3:

```bash
pip install boto3
```

Set env vars:

- `RFSN_PUBLISH_BACKEND=s3`
- `RFSN_PUBLISH_S3_BUCKET=your-bucket`
- `RFSN_PUBLISH_S3_PREFIX=rfsn/runs`

Then:

```bash
export RFSN_SIGNING_KEY="change-me"
python scripts/rfsn_publish_artifacts.py --artifacts artifacts
```

# Append-only audit log

This repo can emit a tamper-evident, append-only audit log in JSONL form.

## What it is

`artifacts/audit_log.jsonl` is a chain of entries. Each entry contains:

- `prev_hash` — the previous entry hash (genesis is 64 zeros)
- `entry_hash` — sha256 over (prev_hash + canonical payload)
- run metadata (run_id, time, repo path, goal, profile, status)
- optional pointers to artifacts (manifest + signature hashes, publish destination)

If any line is removed or edited, verification fails.

## Create an entry

After a run (and after signing/publishing if you use those):

```bash
python scripts/rfsn_audit_append.py --repo /path/to/repo --goal "Fix failing tests" --profile ci_safe --status success
```

## Verify the chain

```bash
python scripts/rfsn_verify_audit_chain.py --log artifacts/audit_log.jsonl
```

## Run IDs

If you don't pass `--run-id`, RFSN derives it from:
- timestamp
- git commit (if available)
- goal
- profile

Format: `<unix>-<profile>-<digest>`

This keeps IDs stable enough for traceability, without leaking secrets.
