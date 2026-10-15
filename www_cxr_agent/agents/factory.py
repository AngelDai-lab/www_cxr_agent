"""Common policy interface for training/inference method variants."""
def create_agent(backend="layered", **kwargs):
    if backend=="layered":
        from .layered_agent import LayeredAgent
        return LayeredAgent(**kwargs)
    if backend=="react_single":
        from .react_single_layer import SingleReactAgent
        return SingleReactAgent(**kwargs)
    if backend=="cot_reflection":
        raise ValueError("Qwen3 text-only policy cannot see image paths. Use a visual CoT baseline via CoTReflectionAgent with the frozen VLM.")
    raise ValueError(f"Unknown backend {backend}")
