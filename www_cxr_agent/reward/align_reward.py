"""Compatibility wrapper using the same report-based reward as training."""
def compute_align_reward(prediction, reference, trajectory, config):
    import asyncio
    import concurrent.futures
    from .reward_judge import evaluate_report_async
    from .rewards import RewardComposer
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        assessment=pool.submit(lambda:asyncio.run(evaluate_report_async(prediction,reference,trajectory))).result(timeout=240)
    reward=RewardComposer(phase=config.get('phase','explore'))
    reward.phase_weights=config.get('w')
    reward.phase_penalties=config.get('phase_penalties')
    reward.set_phase(config.get('phase','explore'))
    if 'lamb_mis' in config:reward.lamb_mis=float(config['lamb_mis'])
    record=dict(trajectory,findings=prediction,report_assessment=assessment)
    return reward.compute_parts({'report':reference},record)['total']
