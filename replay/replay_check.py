import hashlib
import json

def hash_episode(ep: dict) -> str:
    b = json.dumps(ep, sort_keys=True).encode()
    return hashlib.sha256(b).hexdigest()

def verify_replay(original: dict, replayed: dict):
    if hash_episode(original) != hash_episode(replayed):
        raise RuntimeError("Replay divergence detected")
