"""No-layering ablation: free screening inside one shared verification loop."""
from .layered_agent import LayeredAgent, VERIFY_PROMPT

class SingleReactAgent(LayeredAgent):
    async def screen(self):
        self.preprocessed=self.image
        self.preprocessed_lat=self.image_lat

    async def verify(self):
        self.verification_prompt = VERIFY_PROMPT.replace(
            'You are Phase-2 of a chest X-ray (CXR) diagnostic agent',
            'You are a single-loop chest X-ray ReAct agent with no enforced screening stage')
        self.verification_prompt += '\nAlso available: cxr_preprocess(image), cxr_patho_classifier(image). Choose screening calls yourself. The initial H may be empty; discover findings using tools.'
        await super().verify()
