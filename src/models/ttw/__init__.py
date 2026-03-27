"""TTW (Test-Time Warmup) sub-package.

Public entry point: ``TTWModel`` and ``TTW_AUXILIARY_PROMPTS`` from this package.
Implementation modules live under ``src.models.ttw._*``.
"""

from src.models.ttw._config import TTW_AUXILIARY_PROMPTS
from src.models.ttw._strategy import WarmupExecutionProfile, resolve_warmup_execution_profile
from src.models.ttw._wrapper import TTWModel

__all__ = [
    "TTWModel",
    "TTW_AUXILIARY_PROMPTS",
    "WarmupExecutionProfile",
    "resolve_warmup_execution_profile",
]
