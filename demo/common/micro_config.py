"""
Micro kernel configuration for controlling code generation behavior.
Import this module in other micro_*.py files to use these variables.
"""
import os
import tilelang

# =============================================================================
# GPU Architecture Configuration
# =============================================================================
# Target GPU architecture for CUDA code generation
# Supported values: "sm_90a", "sm_89", "sm_80", "sm_86", "sm_89", etc.
TARGET_ARCH = "sm_89"

# =============================================================================
# Megakernel Configuration
# =============================================================================
# Whether to generate megakernel conversion code
# When True, generates additional wrapper code for combining multiple kernels
ENABLE_MEGAKERNEL = True
ENABLE_PROFILING = False

# =============================================================================
# TMA (Tensor Memory Access) Configuration
# =============================================================================
# Enable TMA operations in generated kernels
# ENABLE_TMA = True

# =============================================================================
# Utility Functions
# =============================================================================
def get_target_str():
    """Returns the complete target string for tilelang.jit"""
    return f"cuda -arch={TARGET_ARCH}"

def get_pass_configs():
    return {
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }

# def is_tma_enabled():
#     """Check if TMA should be enabled based on architecture"""
#     # TMA requires sm_90a (Hopper) or newer
#     tma_arches = ["sm_90a", "sm_90", "sm_100", "sm_100a"]
#     return ENABLE_TMA and (TARGET_ARCH in tma_arches)

def is_megakernel_enabled():
    """Check if megakernel mode is enabled"""
    return ENABLE_MEGAKERNEL

def is_enable_profiling():
    """Check if megakernel mode is enabled"""
    return ENABLE_PROFILING