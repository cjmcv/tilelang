#include <tl_templates/cuda/instruction/mma.h>
#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void gemm_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap B_desc, half_t* __restrict__ C);
extern "C" __global__ void __launch_bounds__(256, 1) gemm_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap B_desc, half_t* __restrict__ C) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  __shared__ __align__(16) uint64_t mbarrier_mem[6];
  auto mbarrier = reinterpret_cast<Barrier*>(mbarrier_mem);
  float C_local[128];
  half_t C_local_cast[2];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(A_desc);
    tl::prefetch_tma_descriptor(B_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    mbarrier[0].init(1);
    mbarrier[1].init(1);
    mbarrier[2].init(1);
    mbarrier[3].init(128);
    mbarrier[4].init(128);
    mbarrier[5].init(128);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (128 <= ((int)threadIdx.x)) {
    tl::warpgroup_reg_dealloc<24>();
    for (int k = 0; k < 32; ++k) {
      mbarrier[((k % 3) + 3)].wait((((k % 6) / 3) ^ 1));
      if (tl::tl_shuffle_elect<128>()) {
        mbarrier[(k % 3)].expect_transaction(8192);
        tl::tma_load(A_desc, mbarrier[(k % 3)], (&(((half_t*)buf_dyn_shmem)[((k % 3) * 4096)])), (k * 32), (((int)blockIdx.y) * 128));
        mbarrier[(k % 3)].arrive_and_expect_tx(8192);
        tl::tma_load(B_desc, mbarrier[(k % 3)], (&(((half_t*)buf_dyn_shmem)[(((k % 3) * 4096) + 12288)])), (((int)blockIdx.x) * 128), (k * 32));
        tl::tma_load(B_desc, mbarrier[(k % 3)], (&(((half_t*)buf_dyn_shmem)[(((k % 3) * 4096) + 14336)])), ((((int)blockIdx.x) * 128) + 64), (k * 32));
      }
    }
  } else {
    tl::warpgroup_reg_alloc<240>();
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(C_local + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    for (int k_1 = 0; k_1 < 32; ++k_1) {
      mbarrier[(k_1 % 3)].wait(((k_1 % 6) / 3));
      {
        half_t A_local[32];
        half_t B_local[32];
        for (int ki = 0; ki < 2; ++ki) {
          for (int i_1 = 0; i_1 < 4; ++i_1) {
            tl::ptx_ldmatrix_x4((&(((half_t*)buf_dyn_shmem)[(((((((k_1 % 3) * 4096) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + (i_1 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])) + 0, A_local + (i_1 * 8));
          }
          for (int i_2 = 0; i_2 < 4; ++i_2) {
            tl::ptx_ldmatrix_x4_trans((&(((half_t*)buf_dyn_shmem)[(((((((k_1 % 3) * 4096) + ((((int)threadIdx.x) >> 6) * 2048)) + (ki * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_2 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 12288)])) + 0, B_local + (i_2 * 8));
          }
          for (int i_3 = 0; i_3 < 4; ++i_3) {
            for (int j = 0; j < 4; ++j) {
              tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + ((i_3 * 32) + (j * 8))), reinterpret_cast<const unsigned*>(A_local + (i_3 * 8)), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
              tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + (((i_3 * 32) + (j * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_3 * 8)), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
            }
          }
        }
      }
      mbarrier[((k_1 % 3) + 3)].arrive();
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 64; ++i_4) {
      uint1 __1;
      float2 v_ = *(float2*)(C_local + (i_4 * 2));
      ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
      *(uint1*)(C_local_cast + 0) = __1;
      *(uint1*)(C + (((((((((((int)blockIdx.y) * 131072) + (((((int)threadIdx.x) & 63) >> 5) * 65536)) + ((i_4 >> 4) * 16384)) + ((i_4 & 1) * 8192)) + (((((int)threadIdx.x) & 31) >> 2) * 1024)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) >> 6) * 64)) + (((i_4 & 15) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(C_local_cast + 0);
    }
  }
}


#define ERROR_BUF_SIZE 1024
static char error_buf[ERROR_BUF_SIZE];

extern "C" const char* get_last_error() {
    return error_buf;
}

extern "C" int gemm_init() {
    error_buf[0] = '\0';

    cudaError_t result_gemm_kernel = cudaFuncSetAttribute(gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 49152);
    if (result_gemm_kernel != cudaSuccess) {
        snprintf(error_buf, ERROR_BUF_SIZE, "Failed to set the allowed dynamic shared memory size to %d with error: %s", 49152, cudaGetErrorString(result_gemm_kernel));
        return -1;
    }

    return 0;
}

extern "C" int gemm_call(half_t* __restrict__ A, half_t* __restrict__ B, half_t* __restrict__ C, cudaStream_t stream=cudaStreamDefault) {

        CUtensorMap A_desc;
        CUtensorMapDataType A_desc_type= (CUtensorMapDataType)6;
        cuuint32_t A_desc_tensorRank= 2;
        void *A_desc_globalAddress= A;
        cuuint64_t A_desc_globalDim[2]= {1024,1024};
        cuuint64_t A_desc_globalStride[2]= {2,2048};
        cuuint32_t A_desc_boxDim[2]= {32,128};
        cuuint32_t A_desc_elementStrides[2]= {1,1};
        CUtensorMapInterleave A_desc_interleave= (CUtensorMapInterleave)0;
        CUtensorMapSwizzle A_desc_swizzle= (CUtensorMapSwizzle)2;
        CUtensorMapL2promotion A_desc_l2Promotion= (CUtensorMapL2promotion)2;
        CUtensorMapFloatOOBfill A_desc_oobFill= (CUtensorMapFloatOOBfill)0;

        CUresult A_desc_result = cuTensorMapEncodeTiled(
    &A_desc, A_desc_type, A_desc_tensorRank, A_desc_globalAddress, A_desc_globalDim, A_desc_globalStride + 1, A_desc_boxDim, A_desc_elementStrides, A_desc_interleave, A_desc_swizzle, A_desc_l2Promotion, A_desc_oobFill);

        if (A_desc_result != CUDA_SUCCESS) {
                snprintf(error_buf, ERROR_BUF_SIZE, "Error: Failed to initialize the TMA descriptor A_desc");
                return -1;
        }

        CUtensorMap B_desc;
        CUtensorMapDataType B_desc_type= (CUtensorMapDataType)6;
        cuuint32_t B_desc_tensorRank= 2;
        void *B_desc_globalAddress= B;
        cuuint64_t B_desc_globalDim[2]= {1024,1024};
        cuuint64_t B_desc_globalStride[2]= {2,2048};
        cuuint32_t B_desc_boxDim[2]= {64,32};
        cuuint32_t B_desc_elementStrides[2]= {1,1};
        CUtensorMapInterleave B_desc_interleave= (CUtensorMapInterleave)0;
        CUtensorMapSwizzle B_desc_swizzle= (CUtensorMapSwizzle)3;
        CUtensorMapL2promotion B_desc_l2Promotion= (CUtensorMapL2promotion)2;
        CUtensorMapFloatOOBfill B_desc_oobFill= (CUtensorMapFloatOOBfill)0;

        CUresult B_desc_result = cuTensorMapEncodeTiled(
    &B_desc, B_desc_type, B_desc_tensorRank, B_desc_globalAddress, B_desc_globalDim, B_desc_globalStride + 1, B_desc_boxDim, B_desc_elementStrides, B_desc_interleave, B_desc_swizzle, B_desc_l2Promotion, B_desc_oobFill);

        if (B_desc_result != CUDA_SUCCESS) {
                snprintf(error_buf, ERROR_BUF_SIZE, "Error: Failed to initialize the TMA descriptor B_desc");
                return -1;
        }

        {
                cudaLaunchConfig_t config;
                cudaLaunchAttribute attribute[1];
                attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
                attribute[0].val.programmaticStreamSerializationAllowed = 1;
                config.attrs = attribute;
                config.numAttrs = 1;
                config.stream = stream;
                config.gridDim = dim3(8, 8, 1);
                config.blockDim = dim3(256, 1, 1);
                config.dynamicSmemBytes = 49152;
                cudaLaunchKernelEx(&config, gemm_kernel, A_desc, B_desc, C);
        }

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            snprintf(error_buf, ERROR_BUF_SIZE, "gemm_kernel: %s - %s", cudaGetErrorName(err), cudaGetErrorString(err));
            return -1;
        }

        return 0;
}