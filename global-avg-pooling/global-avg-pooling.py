import numpy as np

def global_avg_pool(x):
    """
    Compute global average pooling over spatial dims.
    Supports (C,H,W) => (C,) and (N,C,H,W) => (N,C).
    """
    # Write code here
    if x.ndim == 3:
        gap = np.mean(x, axis=(1, 2))
    elif x.ndim == 4:
        gap = np.mean(x, axis=(2, 3))
    else:
        raise ValueError("the program doesn't support that tensor shape")
    return gap.astype(np.float64)
    pass