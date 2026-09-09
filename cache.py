import json
import hashlib
import redis
r=redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
def cache_key(question:str):
    return hashlib.md5(question.lower().strip().encode()
                       ).hexdigest()
def get_answer(question:str):
    return r.get(cache_key(question))
def set_answer(question:str, answer:dict):
    r.setex(cache_key(question),3600, json.dumps(answer))


