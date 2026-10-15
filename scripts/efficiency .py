"""Summarize actual recorded latency/tokens; unavailable values remain null."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from www_cxr_agent.eval.metrics_efficiency import compute_efficiency

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--jsonl',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args()
    rows=[json.loads(line) for line in Path(args.jsonl).read_text().splitlines() if line.strip()]
    result=compute_efficiency(rows)
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    Path(args.out).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
