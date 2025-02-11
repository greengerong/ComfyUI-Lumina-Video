import warnings

try:
    from apex.normalization import FusedLayerNorm as LayerNorm, FusedRMSNorm as RMSNorm
except ImportError:
    warnings.warn("Cannot import apex RMSNorm, switch to vanilla implementation")
    from torch.nn import LayerNorm, RMSNorm
