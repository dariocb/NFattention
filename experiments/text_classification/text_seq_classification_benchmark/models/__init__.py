"""
Model implementations for text sequence classification.

Supported models:
- transformer: Standard multi-head self-attention (dot-product attention)
- mikan: Multi-head Implicit Kernel Attention Network (IKAN-direct)
- ours_latest: **当前默认模型** = Hybrid NGSM (no_qk + fixed_orth_v + hybrid architecture + per-head NGSM prior)
- ours_hybrid_noprior: Hybrid + no prior (KL=0)
- ours_hybrid_ngsm:   Hybrid + per-head NGSM prior (= ours_latest 的别名)
- ours_hybrid_rbf:    Hybrid + per-head RBF prior
- ours_shared_ngsm:   旧版 shared_full + NGSM (论文初稿 default，已弃用)
- ours_fixed_qk: Our method with frozen Q, K projections (legacy)
- ours_trainable_qk: Our method with trainable Q, K projections (legacy)
- performer: FAVOR+ linear attention (Choromanski et al., 2020)
- rka: Random Kernel Attention with learned features
- gmm_rks: Gaussian Mixture Model Random Kitchen Sinks
- kpca_scaled: KPCA 论文仓库中的 Scaled / symmetric-K 调制注意力 (Teo & Nguyen, NeurIPS 2024)
- metala: MetaLA 风格因果 GLA token mixer (Chou et al., NeurIPS 2024; 纯 PyTorch 递推)
"""

from .base import BaseClassifier
from .transformer import TransformerClassifier
from .mikan import MIKANClassifier
from .ours import OursClassifier
from .performer import PerformerClassifier
from .rka import RKAClassifier
from .gmm_rks import GMMRKSClassifier
from .kpca_scaled import KPCAScaledClassifier
from .metala_baseline import MetaLABaselineClassifier

__all__ = [
    'BaseClassifier',
    'TransformerClassifier',
    'MIKANClassifier', 
    'OursClassifier',
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
        - "ours_latest": **当前默认** = Hybrid NGSM (no_qk + fixed_orth + hybrid + per-head NGSM prior)
        - "ours_shared_ngsm": 旧版 shared_full + NGSM（初稿 default，保留供消融）
        - "ours_fixed_qk": Our method with frozen Q, K projections (legacy)
        - "ours_trainable_qk": Our method with trainable Q, K projections (legacy)
        - "performer": Performer with FAVOR+ attention
        - "rka": Random Kernel Attention
        - "gmm_rks": GMM-RKS (Gaussian Mixture Random Kitchen Sinks)
        - "kpca_scaled": Scaled attention from KPCA official repo (KPCA_code)
        - "metala": MetaLA-style GLA mixer (PyTorch naive GLA, no flash-linear-attn dep)
    """
    # ── 当前默认配置 ──────────────────────────────────────────────────────────
    # ours_latest = Hybrid NGSM (no_qk + fixed_orth + hybrid arch + per-head NGSM prior)
    # 由 new_experiment/long_seq_classification 实验确定为主推模型。
    # 若要切换 prior，改用 ours_hybrid_noprior 或 ours_hybrid_rbf。
    _LATEST_KWARGS = dict(
        qk_mode='no_qk',
        v_mode='fixed_orth',
        architecture='hybrid',
        prior_type='ngsm',
        trunk_layers=2,
        mlp_hidden_dim=16,
    )

    model_map = {
        # Standard Transformer baseline
        "transformer": TransformerClassifier,

        # MIKAN baseline
        "mikan": MIKANClassifier,

        # MGK baseline
        "mgk": MGKClassifier,

        # ── 当前默认模型：Hybrid NGSM ─────────────────────────────────────────
        # 等价于 ours_hybrid_ngsm，设为 ours_latest 让现有调用脚本无需改动。
        "ours_latest": lambda **kw: OursClassifier(**{**_LATEST_KWARGS, **kw}),
        # New variant: one shared kernel net conditioned on head-id embedding
        "ours_latest_head_kernel": lambda **kw: OursClassifier(
            qk_mode='no_qk',
            v_mode='fixed_orth',
            shared_flow=True,
            use_shared_kernel=True,
            freeze_qk=False,
            **kw
        ),

        # 旧版 shared_full + NGSM（论文初稿 default）— 保留供消融对比
        "ours_shared_ngsm": lambda **kw: OursClassifier(
            qk_mode='no_qk',
            v_mode='fixed_orth',
            shared_flow=True,
            freeze_qk=False,
            **kw
        ),

        # Hybrid 架构 + NoPrior + Tiny NF (新主推)
        # 与 new_experiment/mnist_ablation_tiny_nf 的 FRSKA-Hybrid-NoPrior 对齐
        # 注: num_flows/num_mixtures 由调用者传入（默认 config 已是 3/10），这里不再覆盖
        "ours_hybrid_noprior": lambda **kw: OursClassifier(
            qk_mode='no_qk',
            v_mode='fixed_orth',
            architecture='hybrid',
            prior_type='none',
            trunk_layers=2,
            mlp_hidden_dim=16,
            **kw
        ),

        # Hybrid 架构 + per-head NGSM prior + Tiny NF
        "ours_hybrid_ngsm": lambda **kw: OursClassifier(
            qk_mode='no_qk',
            v_mode='fixed_orth',
            architecture='hybrid',
            prior_type='ngsm',
            trunk_layers=2,
            mlp_hidden_dim=16,
            **kw
        ),

        # Hybrid 架构 + per-head RBF prior (DiagGaussian, trainable σ) + Tiny NF
        "ours_hybrid_rbf": lambda **kw: OursClassifier(
            qk_mode='no_qk',
            v_mode='fixed_orth',
            architecture='hybrid',
            prior_type='rbf',
            trunk_layers=2,
            mlp_hidden_dim=16,
            **kw
        ),

        # ── Deep-trunk 变体 (trunk_layers=4)：5 层 NF = 前 4 层 shared + 1 层 head-specific ──
        # 与上面三个 hybrid 模型唯一区别是 trunk_layers (2→4)。
        # 搭配 config: num_flows=5, flow_hidden_dim=128, M=96 ("latest" 设置)。
        "ours_hybrid_noprior_deep": lambda **kw: OursClassifier(
            qk_mode='no_qk',
            v_mode='fixed_orth',
            architecture='hybrid',
            prior_type='none',
            trunk_layers=4,
            mlp_hidden_dim=16,
            **kw
        ),
        "ours_hybrid_ngsm_deep": lambda **kw: OursClassifier(
            qk_mode='no_qk',
            v_mode='fixed_orth',
            architecture='hybrid',
            prior_type='ngsm',
            trunk_layers=4,
            mlp_hidden_dim=16,
            **kw
        ),
        "ours_hybrid_rbf_deep": lambda **kw: OursClassifier(
            qk_mode='no_qk',
            v_mode='fixed_orth',
            architecture='hybrid',
            prior_type='rbf',
            trunk_layers=4,
            mlp_hidden_dim=16,
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
