"""
Model implementations for text sequence classification.

Supported models:
- transformer: Standard multi-head self-attention (dot-product attention)
- mikan: Multi-head Implicit Kernel Attention Network (IKAN-direct)
- mgk: Mixture of Gaussian Keys baseline
- ours_latest: Latest full setting (no_qk + fixed_orth_v + shared_flow)
- ours_latest_head_kernel: Single shared kernel + head-id embedding (no_qk + fixed_orth_v)
- ours_fixed_qk: Our method with frozen Q, K projections
- ours_trainable_qk: Our method with trainable Q, K projections
- performer: FAVOR+ linear attention (Choromanski et al., 2020)
- rka: Random Kernel Attention with learned features
- gmm_rks: Gaussian Mixture Model Random Kitchen Sinks
- kpca_scaled: KPCA 论文仓库中的 Scaled / symmetric-K 调制注意力 (Teo & Nguyen, NeurIPS 2024)
- metala: MetaLA 风格因果 GLA token mixer (Chou et al., NeurIPS 2024; 纯 PyTorch 递推)
"""

from .base import BaseClassifier
from .transformer import TransformerClassifier
from .mikan import MIKANClassifier
from .mgk import MGKClassifier
from .ours import OursClassifier, OursSharedKernelClassifier
from .performer import PerformerClassifier
from .rka import RKAClassifier
from .gmm_rks import GMMRKSClassifier
from .kpca_scaled import KPCAScaledClassifier
from .metala_baseline import MetaLABaselineClassifier

__all__ = [
    'BaseClassifier',
    'TransformerClassifier',
    'MIKANClassifier', 
    'MGKClassifier',
    'OursClassifier',
    'OursSharedKernelClassifier',
    'PerformerClassifier',
    'RKAClassifier',
    'GMMRKSClassifier',
    'KPCAScaledClassifier',
    'MetaLABaselineClassifier',
    'get_model'
]


def get_model(model_name: str, **kwargs):
    """
    工厂函数：根据名称获取模型
    
    Args:
        model_name: 模型名称
        **kwargs: 模型参数
        
    Returns:
        模型实例
        
    Supported models:
        - "transformer": Standard Transformer with dot-product attention
        - "mikan": MIKAN (Implicit Kernel Attention)
        - "mgk": MGK (Mixture of Gaussian Keys)
        - "ours_latest": Latest full setting (no_qk + fixed_orth_v + shared_flow)
        - "ours_latest_head_kernel": Single shared kernel + head-id embedding
        - "ours_fixed_qk": Our method with frozen Q, K projections
        - "ours_trainable_qk": Our method with trainable Q, K projections
        - "performer": Performer with FAVOR+ attention
        - "rka": Random Kernel Attention
        - "gmm_rks": GMM-RKS (Gaussian Mixture Random Kitchen Sinks)
        - "kpca_scaled": Scaled attention from KPCA official repo (KPCA_code)
        - "metala": MetaLA-style GLA mixer (PyTorch naive GLA, no flash-linear-attn dep)
    """
    model_map = {
        # Standard Transformer baseline
        "transformer": TransformerClassifier,
        
        # MIKAN baseline
        "mikan": MIKANClassifier,
        
        # MGK baseline
        "mgk": MGKClassifier,

        # Our latest full setting used as main model
        "ours_latest": lambda **kw: OursClassifier(
            qk_mode='no_qk',
            v_mode='fixed_orth',
            shared_flow=True,
            freeze_qk=False,
            **kw
        ),

        # New variant: one shared kernel net conditioned on head-id embedding
        "ours_latest_head_kernel": lambda **kw: OursSharedKernelClassifier(
            qk_mode='no_qk',
            v_mode='fixed_orth',
            freeze_qk=False,
            **kw
        ),
        
        # Our method (two variants)
        # Legacy aliases kept for compatibility with old runs.
        "ours_fixed_qk": lambda **kw: OursClassifier(
            qk_mode='normal',
            v_mode='learnable',
            shared_flow=True,
            freeze_qk=True,
            **kw
        ),
        "ours_trainable_qk": lambda **kw: OursClassifier(
            qk_mode='normal',
            v_mode='learnable',
            shared_flow=True,
            freeze_qk=False,
            **kw
        ),
        
        # Performer baseline (FAVOR+)
        "performer": PerformerClassifier,
        
        # RKA baseline (learnable random features)
        "rka": RKAClassifier,
        
        # GMM-RKS baseline (Gaussian Mixture)
        "gmm_rks": GMMRKSClassifier,

        # KPCA (official repo scaled attention)
        "kpca_scaled": KPCAScaledClassifier,

        # MetaLA (structure from BICLab/MetaLA; GLA via naive recurrence)
        "metala": MetaLABaselineClassifier,
    }
    
    if model_name not in model_map:
        available = list(model_map.keys())
        raise ValueError(f"不支持的模型: {model_name}. 可用模型: {available}")
    
    model_class = model_map[model_name]
    if callable(model_class) and not isinstance(model_class, type):
        return model_class(**kwargs)
    return model_class(**kwargs)
