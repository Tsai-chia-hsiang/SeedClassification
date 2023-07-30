import json
import os

def revers_dict(d:dict):
    return {v:k for k,v in d.items()}

def writejson(o,filepath:os.PathLike):
    with open(filepath, "w+", encoding="utf-8") as f:
        json.dump(o, fp=f, indent=4, ensure_ascii=False)
        
def readjson(filepath:os.PathLike):
    ret = None
    with open(filepath, "r", encoding="utf-8") as d:
        ret = json.load(d)
    return ret

def combine_key_value_list(keylst:list, vallst:list)->dict:
    assert len(keylst) == len(vallst)
    return {k:v for k,v in zip(keylst, vallst)}
