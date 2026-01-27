class Qwen3MlpConfig:
    rmsnorm_layout = (32, 1, 1), (1, 1, 1)
    linear1_layout = (96, 2, 1), (64, 16, 64)
    silu_mul_layout = (48, 2, 1), (64, 16, 1)
    linear2_layout = (16, 2, 1), (64, 16, 128)
