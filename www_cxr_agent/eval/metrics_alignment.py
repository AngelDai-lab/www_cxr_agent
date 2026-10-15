"""Report-based alignment, separate abnormal support and total finding coverage."""
def compute_alignment(records):
    if not records:
        raise ValueError("Nonempty evaluation cohort required")
    assessments=[r.get("trajectory",{}).get("report_assessment") for r in records]
    if any(a is None for a in assessments):
        raise ValueError("Independent report assessment required for every record")
    candidate=sum(a["candidate_count"] for a in assessments)
    unsupported=sum(a["unsupported_count"] for a in assessments)
    all_count=sum(a["all_finding_count"] for a in assessments)
    supported=sum(a["supported_finding_count"] for a in assessments)
    coverage=[]
    completed=0
    for r in records:
        items=r.get("trajectory",{}).get("checklist_checked",[])
        completed+=r.get("trajectory",{}).get("termination",{}).get("reason")=="complete"
        coverage.append(len({it["name"] for it in items if it.get("status")=="checked" and it.get("evidence")})/7)
    return {"interpretation":"Trace support, not image-level hallucination; no-tool baselines have no trace evidence",
            "misalign_rate":sum(a["misalign"] for a in assessments)/len(assessments),
            "misalign_rate_micro":unsupported/candidate if candidate else 0.,
            "evidence_coverage":supported/all_count if all_count else None,
            "checklist_coverage":sum(coverage)/len(coverage),
            "complete_termination_rate":completed/len(records),
            "candidate_abnormal_count":candidate,"unsupported_abnormal_count":unsupported}
