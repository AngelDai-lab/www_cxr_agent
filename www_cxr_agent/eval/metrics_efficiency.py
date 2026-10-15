"""Measured inference/tool costs; missing measurements stay unavailable."""
import json

def compute_efficiency(records):
    n=len(records)
    if not n:
        raise ValueError("No records")
    calls=[tc for r in records for tc in r.get("trajectory",{}).get("tool_calls",[])]
    duplicates=0
    for record in records:
        seen=set()
        for tc in record.get("trajectory",{}).get("tool_calls",[]):
            key=json.dumps([tc["name"],tc.get("arguments",{})],sort_keys=True)
            duplicates+=key in seen;seen.add(key)
    used=sum(len({i for i in r.get("trajectory",{}).get("used_tool_indices",[]) if isinstance(i,int) and 0<=i<len(r.get("trajectory",{}).get("tool_calls",[]))}) for r in records)
    result={"mean_tool_calls":len(calls)/n,
            "referenced_tool_rate":used/len(calls) if calls else None,
            "call_success_rate":sum(tc.get("status")=="ok" for tc in calls)/len(calls) if calls else None,
            "exact_duplicate_call_rate":duplicates/len(calls) if calls else None,
            "mean_tool_seconds":sum(tc.get("duration",0) for tc in calls)/n}
    for name in ("elapsed_seconds","prompt_tokens","completion_tokens"):
        values=[r.get("efficiency",{}).get(name) for r in records]
        result["mean_"+name]=sum(values)/n if all(v is not None for v in values) else None
    return result
