class Qwen3MlpConfig:
    rmsnorm_layout = (1, 1, 1), (1, 1, 1)
    linear1_layout = (304, 1, 1), (64, 16, 64)
    silu_mul_layout = (152, 1, 1), (64, 16, 1)
    linear2_layout = (40, 1, 1), (64, 16, 64)
