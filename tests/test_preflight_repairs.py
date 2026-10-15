import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from www_cxr_agent.tools.llm_judge_tool import CxrNegationCheckerTool
from www_cxr_agent.train.grpo_trainer import GRPOTrainer

class Repairs(unittest.TestCase):
    def test_strict_json(self):
        parse=CxrNegationCheckerTool.parse_json_object
        self.assertEqual(parse('```json\n{"score": 1}\n```'), {'score':1})
        self.assertEqual(parse('{"text":"a {brace} and \\"quote\\""}'), {'text':'a {brace} and "quote"'})
        for text in ['{"score":', '{"score":1}{"score":0}', 'no JSON']:
            with self.assertRaises(ValueError): parse(text)

    def test_validation_reuses_complete_predictions_without_model(self):
        async def evaluate(records):
            self.assertEqual(records[0]['id'],'case')
            return {'F1_14':0.5,'complete':True}
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'validation_step40.jsonl'
            p.write_text(json.dumps({'id':'case','validation_checkpoint_step':40})+'\n')
            trainer=object.__new__(GRPOTrainer)
            with patch('www_cxr_agent.eval.runner.evaluate_records_async',evaluate):
                result=asyncio.run(trainer.validate([{'id':'case'}],d,40))
            self.assertEqual(result['F1_14'],0.5)
            self.assertTrue((Path(d)/'validation_step40.json').exists())

if __name__=='__main__': unittest.main()
