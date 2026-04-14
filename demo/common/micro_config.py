"""
Micro kernel configuration for controlling code generation behavior.
Import this module in other micro_*.py files to use these variables.
"""

# =============================================================================
# GPU Architecture Configuration
# =============================================================================
# Target GPU architecture for CUDA code generation
# Supported values: "sm_90a", "sm_89", "sm_80", "sm_86", "sm_89", etc.
TARGET_ARCH = "sm_90a"

# =============================================================================
# Megakernel Configuration
# =============================================================================
# Whether to generate megakernel conversion code
# When True, generates additional wrapper code for combining multiple kernels
ENABLE_MEGAKERNEL = False

# =============================================================================
# TMA (Tensor Memory Access) Configuration
# =============================================================================
# Enable TMA operations in generated kernels
ENABLE_TMA = True

ENABLE_PROFILING = False


# =============================================================================
# Utility Functions
# =============================================================================
def get_target_str():
    """Returns the complete target string for tilelang.jit"""
    return f"cuda -arch={TARGET_ARCH}"

def is_tma_enabled():
    """Check if TMA should be enabled based on architecture"""
    # TMA requires sm_90a (Hopper) or newer
    tma_arches = ["sm_90a", "sm_90", "sm_100", "sm_100a"]
    return ENABLE_TMA and (TARGET_ARCH in tma_arches)

def is_megakernel_enabled():
    """Check if megakernel mode is enabled"""
    return ENABLE_MEGAKERNEL

def is_enable_profiling():
    """Check if megakernel mode is enabled"""
    return ENABLE_PROFILING