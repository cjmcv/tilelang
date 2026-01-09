#ifndef MEGAKERNEL_USE_CUTLASS_KERNEL
// Mirage use a flag (use_cutlass_kernel) to control whether use the cutlass
// version kernel or not.
#define MEGAKERNEL_USE_CUTLASS_KERNEL 1
#endif // MEGAKERNEL_USE_CUTLASS_KERNEL

#include "argmax.cuh"
#include "embedding.cuh"
#include "identity.cuh"
#include "multitoken_paged_attention.cuh"
#include "reduction.cuh"
#include "rmsnorm.cuh"
#include "rotary_embedding.cuh"
#include "silu_mul.cuh"

#if MEGAKERNEL_USE_CUTLASS_KERNEL
// #include "linear_cutlass.cuh"
// #include "linear_cutlass_withp.cuh"
// #include "linear_cutlass_origin.cuh"
#include "linear_gemv_cutlass.cuh"
#else
#include "linear.cuh"
#endif // MEGAKERNEL_USE_CUTLASS_KERNEL