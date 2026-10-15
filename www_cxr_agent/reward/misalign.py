"""Independent report/evidence assessment compatibility entrypoint."""
def compute_misalign(prediction, evidence_cache):
    import asyncio
    import concurrent.futures
    from .reward_judge import evaluate_report_async
    # A cache without original tool calls cannot establish verified support.
    if not isinstance(evidence_cache,dict) or 'tool_calls' not in evidence_cache:
        raise ValueError('Full trajectory with original tool calls is required')
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        assessment=pool.submit(lambda:asyncio.run(evaluate_report_async(prediction,'',evidence_cache))).result(timeout=240)
    return assessment['misalign']
