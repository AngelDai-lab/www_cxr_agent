"""Run comparable method/ablation jobs; execution requires explicit --execute.

External published baselines require actual predictions through the strict adapter.
The matrix is generated in memory; no plan/report files are produced.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from train_grpo import load_cfg

def variants():
    base=load_cfg('configs/train_default.yaml')
    import copy
    result={}
    def add(name,change):
        cfg=copy.deepcopy(base);change(cfg)
        cfg['paths']={'runs_dir':f'runs/{name}','checkpoint_dir':f'runs/{name}/checkpoints'}
        cfg['validation']={'enabled':True,'n':50,'interval':10,'patience':3}
        result[name]=cfg
    add('full',lambda c:None)
    add('no_grpo',lambda c:c.update())
    add('no_layering',lambda c:c.update(agent_backend='react_single'))
    add('no_checklist',lambda c:c.update(agent_options={'no_checklist':True}))
    add('single_view',lambda c:c.update(single_view=True,paired_only=True))
    add('dual_view',lambda c:c.update(paired_only=True))
    add('no_disambig',lambda c:c.update(agent_options={'no_disambig':True}))
    for name,index in [('succ',0),('div',1),('coh',2),('judge',3)]:
        def remove(c,index=index):
            for weights in c['reward']['w'].values():weights[index]=0.
        add('no_r'+name,remove)
    add('no_rtool',lambda c:c['reward'].update(w={'explore':[0.,0.,0.,0.],'converge':[0.,0.,0.,0.]}))
    for value in [0.,.1,.2,.3,.5]:
        add('lambda_'+str(value),lambda c,v=value:c['reward'].update(phase_penalties={'explore':v,'converge':v},lamb_mis=v))
    for component,index in [('succ',0),('div',1),('coh',2),('judge',3)]:
        for value in [.1,.3,.5]:
            def weight(c,i=index,v=value):
                for weights in c['reward']['w'].values():weights[i]=v
            add(f'weight_{component}_{value}',weight)
    add('converge_only',lambda c:c['reward']['w'].update(explore=list(c['reward']['w']['converge'])))
    result['converge_only']['reward']['phase_penalties']={'explore':.3,'converge':.3}
    add('reverse_curriculum',lambda c:c['reward'].update(w={'explore':[.1,.2,.2,.2],'converge':[.1,.5,.5,0.]},phase_penalties={'explore':.3,'converge':.2}))
    for size in [4,8,14]:
        add(f'qwen3_{size}b',lambda c,v=size:c['policy'].update(model=f'/root/autodl-tmp/models/Qwen3-{v}B-Instruct'))
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--variant',default='full')
    ap.add_argument('--execute',action='store_true')
    ap.add_argument('--n',type=int)
    ap.add_argument('--dataset',choices=['mimic_cxr','iu_xray'],default='mimic_cxr')
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--curriculum-step',type=int,default=30)
    args=ap.parse_args();matrix=variants()
    if args.variant not in matrix:raise ValueError('Variants: '+', '.join(matrix))
    cfg=matrix[args.variant]
    cfg['data'].update(dataset=args.dataset,seed=args.seed)
    tag=f'{args.variant}_{args.dataset}_seed{args.seed}'
    cfg['paths']={'runs_dir':f'runs/{tag}','checkpoint_dir':f'runs/{tag}/checkpoints'}
    if not args.execute:
        print(yaml.safe_dump(cfg,sort_keys=False));return
    if not Path(cfg['policy']['model']).is_dir():raise FileNotFoundError(cfg['policy']['model'])
    root=Path(cfg['paths']['runs_dir']);root.mkdir(parents=True,exist_ok=True)
    config=root/'config.yaml';config.write_text(yaml.safe_dump(cfg,sort_keys=False))
    # Respect the launcher's CUDA_VISIBLE_DEVICES. The old hard-coded ``0``
    # silently disabled the four-card deployment for every matrix experiment.
    env = dict(os.environ)
    env.setdefault('CUDA_VISIBLE_DEVICES', '0,1,2,3')
    if args.variant == 'no_grpo':
        pred = root / 'predictions.jsonl'
        metrics = root / 'metrics.json'
        cmd=[sys.executable,'scripts/infer.py','--phase','full','--backend','layered',
             '--out',str(pred),'--dataset',args.dataset]
        if args.n: cmd += ['--n',str(args.n)]
        subprocess.run(cmd,check=True,env=env)
        subprocess.run([sys.executable,'scripts/eval.py','--jsonl',str(pred),'--out',str(metrics)],check=True,env=env)
        return
    cmd=[sys.executable,'scripts/train_grpo.py','--config',str(config),'--phase',f'full@{args.curriculum_step}']
    if args.n:cmd+=['--n',str(args.n)]
    subprocess.run(cmd,check=True,env=env)
if __name__=='__main__':main()
