"""Read-only training metrics exporter; --watch updates without restarting training."""
import argparse
import csv
import json
import time
from pathlib import Path
from statistics import mean

REWARD = {'f1_14':'train_F1_14_sample_mean', 'f1_abnorm':'train_F1_abnorm',
          'r_misalign':'train_P_misalign', 'quality':'train_quality',
          'r_tool':'train_R_tool', 'lambda_mis':'lambda_mis',
          'misalign_penalty':'train_misalign_penalty', 'total':'train_reward_exact'}
TOOLS = {k:'train_'+v for k,v in [('r_succ','R_succ'),('r_div','R_div'),('r_coh','R_coh'),('r_judge','R_toolJudge')]}
VAL = ['F1_14','F1_abnorm','FP_rate','Miss_rate','P_misalign','R_succ','R_div','R_coh',
       'checklist_coverage','evidence_coverage','tool_redundancy_rate']

def export(root, cache):
    with (root/'step_stats.csv').open(newline='') as f:
        reader=csv.DictReader(f); base=list(reader.fieldnames); rows=list(reader)
    for row in rows:
        step=int(row['step']); path=root/f'trajectories_per_step/step{step}.jsonl'
        if path.exists():
            stamp=(path.stat().st_mtime_ns,path.stat().st_size)
            if step not in cache or cache[step][0]!=stamp:
                vals={v:[] for v in list(REWARD.values())+list(TOOLS.values())}; phases=set();versions=set();n=0
                with path.open() as f:
                    for line in f:
                        t=json.loads(line)['trajectory']; n+=1
                        rp=t.get('reward_parts',{}); phases.add(rp.get('reward_phase',''));versions.add(rp.get('reward_impl_version',''))
                        for source,mapping in [(rp,REWARD),(t.get('tool_reward_parts',{}),TOOLS)]:
                            for key,col in mapping.items():
                                if key in source: vals[col].append(float(source[key]))
                summary={k:mean(v) if len(v)==n and n else '' for k,v in vals.items()}
                summary.update(reward_phase='|'.join(sorted(phases)),reward_impl_version='|'.join(sorted(versions)),trajectory_count=n)
                cache[step]=(stamp,summary)
            row.update(cache[step][1])
        vpath=root/f'checkpoints/validation_step{step}.json'
        row['validation_complete']=''
        if vpath.exists():
            v=json.loads(vpath.read_text());row['validation_complete']=v.get('complete',False);row['val_n']=v.get('n','')
            if v.get('complete'):
                row.update({'val_'+k:v.get(k,'') for k in VAL})
    fields=base+['reward_phase','reward_impl_version','trajectory_count']+list(REWARD.values())+list(TOOLS.values())+['validation_complete','val_n']+['val_'+k for k in VAL]
    out=root/'metrics_summary.csv';tmp=out.with_suffix('.csv.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    tmp.replace(out)
    return len(rows)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run_dir',type=Path);p.add_argument('--watch',action='store_true');a=p.parse_args();cache={}
    while True:
        try: print('exported',export(a.run_dir,cache),'steps',flush=True)
        except (OSError,ValueError,KeyError) as e:
            if not a.watch: raise
            print('source incomplete; retry:',repr(e),flush=True)
        if not a.watch: break
        time.sleep(60)
