"""
Micro kernel configuration for controlling code generation behavior.
Import this module in other micro_*.py files to use these variables.
"""
import os
import tilelang

TEST_TEMP_HOPPER = True

if TEST_TEMP_HOPPER == True:
    TARGET_ARCH = "sm_90" # "sm_120"
    ENABLE_MEGAKERNEL = False
    PASS_CONFIGS = {
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
else:
    TARGET_ARCH = "sm_89"
    ENABLE_MEGAKERNEL = True
    PASS_CONFIGS = {
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }

ENABLE_PROFILING = False

# =============================================================================
# Utility Functions
# =============================================================================
def get_target_str():
    """Returns the complete target string for tilelang.jit"""
    return f"cuda -arch={TARGET_ARCH}"

def get_arch():
    return TARGET_ARCH

def get_pass_configs():
    return PASS_CONFIGS

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