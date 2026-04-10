# Compatibility shim for submodules that import from src.eval.
from eval.mteb_hf_patches import apply_mteb_hf_patches  # noqa: F401
from eval import run_mmteb  # noqa: F401
