"""Formal report evaluation; unavailable metrics raise instead of becoming zero."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from www_cxr_agent.eval.runner import run_eval

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--jsonl',required=True)
    ap.add_argument('--references')
    ap.add_argument('--out',required=True)
    args=ap.parse_args()
    result=run_eval(args.jsonl,args.references)
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    Path(args.out).write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
