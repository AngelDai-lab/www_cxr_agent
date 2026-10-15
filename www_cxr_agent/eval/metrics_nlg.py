"""Corpus BLEU and mean ROUGE-L/METEOR; missing resources are errors."""
def compute_nlg(predictions, references):
    import nltk
    from pathlib import Path
    nltk.data.path.insert(0,str(Path(__file__).resolve().parents[2]/"cache/nltk"))
    from nltk.translate.bleu_score import corpus_bleu
    from nltk.translate.meteor_score import meteor_score
    from rouge_score.rouge_scorer import RougeScorer
    if not predictions or len(predictions)!=len(references):
        raise ValueError("Nonempty aligned reports required")
    p=[s.lower().split() for s in predictions];r=[s.lower().split() for s in references]
    scorer=RougeScorer(["rougeL"],use_stemmer=True)
    n=len(p)
    # NLTK unsmoothed BLEU uses the smallest float for zero n-gram matches;
    # expose mathematical zero rather than tiny nonzero pseudo-precision.
    bleu4=float(corpus_bleu([[x] for x in r],p,weights=(.25,.25,.25,.25)))
    from collections import Counter
    def grams(tokens):return Counter(tuple(tokens[i:i+4]) for i in range(max(0,len(tokens)-3)))
    if not sum(sum((grams(pred)&grams(ref)).values()) for pred,ref in zip(p,r)):bleu4=0.0
    return {"bleu1":float(corpus_bleu([[x] for x in r],p,weights=(1,0,0,0))),
            "bleu4":bleu4,
            "rouge_l":sum(scorer.score(ref,pred)["rougeL"].fmeasure for pred,ref in zip(predictions,references))/n,
            "meteor":sum(meteor_score([ref],pred) for pred,ref in zip(p,r))/n}
