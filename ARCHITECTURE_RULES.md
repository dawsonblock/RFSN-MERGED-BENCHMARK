Authority rules:

- rfsn_controller/ is the only executor
- gate_ext/ must be pure
- learner/ must be offline
- planner/ may not execute
- CI is the only automated trainer
- replay must hash-match or fail

If a file violates this, delete it.
