import numpy as np

def load_gradient_table(filename):
    with open(filename) as f:
        lines = [
            line.strip() for line in f
            if line.strip() and not line.startswith('##')
        ]
    return np.array([float(line) for line in lines])


# def k_space_from_gradients(*args, **kwargs):
#     raise RuntimeError(
#         "k_space_from_gradients is NOT valid for SPRITE acquisitions. "
#         "Use direct k-space indexing instead."
#     )


