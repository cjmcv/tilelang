class Qwen3MegaConfig:
    rmsnorm_layout = (1, 1, 1), (1, 1, 1)
    linear1_layout = (96, 1, 1), (64, 16, 64)
    silu_mul_layout = (48, 1, 1), (64, 16, 1)
    linear2_layout = (16, 1, 1), (64, 16, 128)
    qkv_proj_layout = (64, 1, 1), (64, 16, 128)
    rope_layout = (24, 1, 1), (1, 1, 1)
    gqa_decode_layout = (1, 8, 4), (64, 64, 4), (16, 1, 1), (64, 64, 4)
    o_proj_layout = (16, 1, 1), (64, 16, 128)
