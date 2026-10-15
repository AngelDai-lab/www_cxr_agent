"""Rollout trajectory schema and helpers."""


class Trajectory:
    def __init__(self):
        self.tool_calls = []
        self.evidence_cache = []
        self.checklist_checked = []
        self.findings = ""

    def to_dict(self):
        return {
            "tool_calls": self.tool_calls,
            "evidence_cache": self.evidence_cache,
            "checklist_checked": self.checklist_checked,
            "findings": self.findings,
        }
