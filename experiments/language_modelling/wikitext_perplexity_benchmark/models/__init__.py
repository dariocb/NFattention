"""Language modeling model registry."""

from .transformer import TransformerLM
from .mikan import MIKANLM
from .performer import PerformerLM
from .rka import RKALM
from .gmm_rks import GMMRKSLM
from .ours import OursLM

__all__ = [
    "TransformerLM",
    "MIKANLM",
    "PerformerLM",
    "RKALM",
    "GMMRKSLM",
    "OursLM",
    "get_model",
]


def get_model(model_name: str, **kwargs):
    model_map = {
        "transformer": TransformerLM,
        "mikan": MIKANLM,
        "performer": PerformerLM,
        "rka": RKALM,
        "gmm_rks": GMMRKSLM,
        "ours_latest": lambda **kw: OursLM(architecture="hybrid", prior_type="ngsm", **kw),
        "ours_shared_ngsm": lambda **kw: OursLM(architecture="shared_full", prior_type="ngsm", **kw),
        "ours_hybrid_noprior": lambda **kw: OursLM(architecture="hybrid", prior_type="none", **kw),
        "ours_hybrid_ngsm": lambda **kw: OursLM(architecture="hybrid", prior_type="ngsm", **kw),
        "ours_hybrid_rbf": lambda **kw: OursLM(architecture="hybrid", prior_type="rbf", **kw),
    }
    if model_name not in model_map:
        raise ValueError(f"Unsupported model: {model_name}. Available: {list(model_map.keys())}")
    model_cls = model_map[model_name]
    return model_cls(**kwargs) if callable(model_cls) and not isinstance(model_cls, type) else model_cls(**kwargs)

