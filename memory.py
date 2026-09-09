from __future__ import annotations
import json
try:
    import redis

    r = redis.Redis(
        host="localhost",
        port=6379,
        db=0
    )
except Exception:
    r=None
MAX_HISTORY=20
def add_message(
    session_id: str,
    role: str,
    content: str
):
    if r is None:
        return
    r.rpush(
        f"chat:{session_id}",
        json.dumps({
            "role": role,
            "content": content
        })
    )
def get_history(
    session_id: str
):
    if r is None:
        return[]
    messages = r.lrange(
        f"chat:{session_id}",
        0,
        -1
    )
    return [
        json.loads(msg)
        for msg in messages
    ]