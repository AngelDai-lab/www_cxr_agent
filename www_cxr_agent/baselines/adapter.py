"""Strict conversion of external baseline predictions; no fabricated reports."""
import json
from pathlib import Path

def adapt_baseline_predictions(input_path, output_path, method_name):
    rows=[json.loads(line) for line in Path(input_path).read_text().splitlines() if line.strip()]
    seen=set();records=[]
    for row in rows:
        key=str(row["id"])
        if key in seen:raise ValueError("Duplicate baseline sample ID: "+key)
        seen.add(key)
        prediction=row.get("findings",row.get("prediction"))
        reference=row.get("report_ref",row.get("reference"))
        if not prediction or not reference:raise ValueError("Baseline requires prediction and reference")
        records.append(dict(id=key,findings=prediction,report_ref=reference,method=method_name,
                            trajectory=row.get("trajectory",{}),efficiency=row.get("efficiency",{})))
    Path(output_path).parent.mkdir(parents=True,exist_ok=True)
    Path(output_path).write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in records))
    return records
