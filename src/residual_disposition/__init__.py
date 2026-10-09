"""Auditing suppressed-but-recoverable model organisms: current behavior or residual disposition?"""

from .paths import load_env

# First, before any module imports huggingface_hub, which reads HF_HOME and HF_TOKEN on import.
load_env()
