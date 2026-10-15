"""Frozen CheXbert report labels and official RadGraph metrics; no proxy fallback."""
import os
from functools import lru_cache
from pathlib import Path

LABELS = ["Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
          "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
          "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", "No Finding"]

class ReportClassifier:
    def __init__(self, root=None, device="cpu"):
        import torch
        from transformers import BertConfig, BertModel, BertTokenizer
        root = Path(root or os.environ.get("CXR_CHEXBERT_DIR", "cache/chexbert"))
        checkpoint = root / "chexbert.pth"
        if not checkpoint.is_file() or not (root / "vocab.txt").is_file():
            raise FileNotFoundError(f"Official CheXbert checkpoint and bert-base-uncased vocab.txt required in {root}")
        class Labeler(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bert = BertModel(BertConfig())
                self.linear_heads = torch.nn.ModuleList([torch.nn.Linear(768, 4) for _ in range(13)] + [torch.nn.Linear(768, 2)])
            def forward(self, **inputs):
                cls = self.bert(**inputs).last_hidden_state[:, 0]
                return torch.stack([head(cls).argmax(-1) for head in self.linear_heads], dim=1)
        self.model = Labeler()
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = state.get("model_state_dict", state)
        self.model.load_state_dict({k.removeprefix("module."): v for k,v in state.items()}, strict=True)
        self.model.to(device).eval().requires_grad_(False)
        import hashlib
        with checkpoint.open("rb") as stream:
            self.checkpoint_sha256=hashlib.file_digest(stream,"sha256").hexdigest()
        self.tokenizer = BertTokenizer(vocab_file=str(root / "vocab.txt"), do_lower_case=True)
        self.device = device

    def predict(self, reports):
        import torch
        with torch.inference_mode():
            inputs = self.tokenizer(reports, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
            raw = self.model(**inputs).cpu().tolist()
        # CheXbert encoding: blank=0, positive=1, negative=2, uncertain=3.
        # Positive-only CE convention; uncertainty remains available in raw labels.
        return [[int(v == 1) for v in row] for row in raw], raw

@lru_cache(maxsize=1)
def report_classifier():
    return ReportClassifier()

def report_quality(prediction, reference):
    from sklearn.metrics import f1_score
    labels, raw = report_classifier().predict([prediction, reference])
    p,r = labels
    # Standard multilabel 14-class macro-F1: calculate one binary F1 per
    # CheXbert class, including zero scores for classes with no positive match.
    f1_14 = float(f1_score([r], [p], average="macro", zero_division=0))
    return {"f1_14": f1_14,
            "report_labels": p, "reference_labels": r, "chexbert_raw": raw,
            "label_order": LABELS, "uncertainty_policy": "positive_only",
            "classifier_sha256":report_classifier().checkpoint_sha256}

def compute_clinical(predictions, references, include_radgraph=True):
    import numpy as np
    from sklearn.metrics import f1_score, precision_score, recall_score
    if len(predictions) != len(references) or not predictions:
        raise ValueError("Nonempty aligned predictions/references required")
    labels,_ = report_classifier().predict(list(predictions)+list(references))
    p,r = np.array(labels[:len(predictions)]),np.array(labels[len(predictions):])
    result = {f"ce_{metric}_{avg}": float(fn(r,p,average=avg,zero_division=0))
              for metric,fn in [("f1",f1_score),("precision",precision_score),("recall",recall_score)]
              for avg in ["micro","macro"]}
    per_fp=((p==1)&(r==0)).sum(axis=0); per_tn=((p==0)&(r==0)).sum(axis=0)
    per_fn=((p==0)&(r==1)).sum(axis=0); per_tp=((p==1)&(r==1)).sum(axis=0)
    fpr=np.divide(per_fp,per_fp+per_tn,out=np.zeros(14,dtype=float),where=(per_fp+per_tn)>0)
    fnr=np.divide(per_fn,per_fn+per_tp,out=np.zeros(14,dtype=float),where=(per_fn+per_tp)>0)
    result.update(fp_rate_macro14=float(fpr.mean()),miss_rate_macro14=float(fnr.mean()),
                  rate_zero_denominator_policy="zero",per_label_fp_rate=fpr.tolist(),per_label_miss_rate=fnr.tolist())
    fp,tn,fn,tp = [int(x.sum()) for x in [(p==1)&(r==0),(p==0)&(r==0),(p==0)&(r==1),(p==1)&(r==1)]]
    result["classifier_sha256"]=report_classifier().checkpoint_sha256
    result["label_order"]=LABELS
    result["macro_zero_division"]=0
    result.update(fp=fp,tn=tn,fn=fn,tp=tp,fp_rate=fp/(fp+tn) if fp+tn else None,
                  miss_rate=fn/(fn+tp) if fn+tp else None,uncertainty_policy="positive_only")
    if include_radgraph:
        from radgraph import F1RadGraph
        import torch
        metric = F1RadGraph(reward_level="all", model_type="radgraph", cuda=-1,
                            model_cache_dir=os.environ.get("CXR_RADGRAPH_DIR", "cache/radgraph"))
        with torch.inference_mode():
            means,_,_,_ = metric(hyps=list(predictions),refs=list(references))
        result.update({f"radgraph_{key}_f1":float(v) for key,v in zip(["simple","partial","complete"],means)})
    return result
