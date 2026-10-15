"""Dataset loaders (raw image paths only; preprocessing is MCP ``cxr_preprocess``)."""

from www_cxr_agent.data.checklist import load_checklist
from www_cxr_agent.data.iu_xray import IUXRayDataset, build_iu_xray_samples
from www_cxr_agent.data.mimic_cxr import MimicCXRDataset, build_mimic_cxr_samples
from www_cxr_agent.data.schema import SampleRecord

__all__ = [
    "SampleRecord",
    "MimicCXRDataset",
    "build_mimic_cxr_samples",
    "IUXRayDataset",
    "build_iu_xray_samples",
    "load_checklist",
]
