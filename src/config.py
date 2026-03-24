from pathlib import Path
import os

"""Central configuration for the MRI reconstruction package."""

corruption_strategy = "random_dropout_corruption"

MASKING_CORRUPTIONS = {
    "low_frequency_corruption",
    "high_frequency_corruption",
    "random_dropout_corruption",
}

# Data
data_root = Path("datasets")
train_split = 0.9
batch_size = 64
# Keep a couple CPU cores free so training and the OS stay responsive.
num_workers = max(2, min(8, (os.cpu_count() or 4) - 2))
pin_memory = True
prefetch_factor = 4
persistent_workers = True
seed = 42

# Normalization
normalization_mode = "max_abs"  # max_abs | std | none
normalization_epsilon = 1e-8

# Corruption parameters
low_frequency_fraction = 0.18
high_frequency_center_fraction = 0.22
random_dropout_probability = 0.55
noise_sigma = 0.03
include_noise_map = True

# Model
base_channels = 32
bilinear_upsampling = True

# Optimization
epochs = 80
learning_rate = 2e-4
weight_decay = 1e-4
max_grad_norm = 1.0
scheduler_name = "cosine"  # cosine | step | none
step_size = 20
gamma = 0.5

# Loss
kspace_loss_type = "l1"  # l1 | mse
image_loss_type = "l1"  # l1 | mse
lambda_k = 1.0
lambda_img = 1.0
ssim_weight = 0.0

# Data consistency
apply_data_consistency = True

# Runtime
device = "auto"  # auto | cpu | cuda

# Output
output_root = Path("outputs")
checkpoint_dir = output_root / "checkpoints"
figure_dir = output_root / "figures"
visualize_every = 10


def is_masking_corruption(strategy: str) -> bool:
    """Return whether a strategy represents missing-coefficient reconstruction."""
    return strategy in MASKING_CORRUPTIONS


def get_corruption_params(strategy: str) -> dict:
    """Map corruption strategy to its parameter dictionary."""
    if strategy == "low_frequency_corruption":
        return {"center_fraction": low_frequency_fraction}
    if strategy == "high_frequency_corruption":
        return {"center_fraction": high_frequency_center_fraction}
    if strategy == "random_dropout_corruption":
        return {"dropout_probability": random_dropout_probability}
    if strategy == "additive_complex_noise":
        return {"sigma": noise_sigma, "return_noise_map": include_noise_map}
    raise ValueError(f"Unsupported corruption strategy: {strategy}")
