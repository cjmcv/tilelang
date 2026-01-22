import itertools

import tilelang
import tilelang.language as T

from common.micro_base import BaseMicroKernel, HparamSelectMode
    
class _SiluMulStrategy:
    def __init__(self, M, N, dtype, accum_dtype):
        self.name = "silu_mul_tl"+f"_{M}_{N}"
            
        self.M = M
        self.N = N
        self.dtype = dtype
        self.accum_dtype = accum_dtype
        
        self.hparam_space = self._get_hparam_space()
        print(len(self.hparam_space))
        
    def _get_hparam_space(self):
        BLOCK_M=[32] #, 256
        BLOCK_N=[64] # , 128
        thread_nums=[128]#
        
        res = []
        for m, n, thread_num in itertools.product(
           BLOCK_M, BLOCK_N, thread_nums):
            res.append([m, n, thread_num])
        return res 
    
    def get_heuristic_hparams(self):
        # [BLOCK_M, BLOCK_N, threads]
        return [32,64,128]
        
    def get_kernel(self, selected_hparams):
        print("selected_hparams: ", selected_hparams)
        return self.kernel_main(self.M, self.N, *selected_hparams, self.dtype, self.accum_dtype) 

    @tilelang.jit(out_idx=[-1])
    def kernel_main(M, N, BLOCK_M, BLOCK_N, threads, dtype="bfloat16", accum_dtype="float32"):
        @T.prim_func
        def silu_mul(
            A: T.Tensor((M, N*2), dtype),
            C: T.Tensor((M, N), dtype),
        ):
            with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M, BLOCK_M), threads=threads) as (bx, by):
                n_block_num = T.ceildiv(N, BLOCK_N)
                # shared tile
                A_sh = T.alloc_shared((BLOCK_M, BLOCK_N), dtype)
                B_sh = T.alloc_shared((BLOCK_M, BLOCK_N), dtype)
                C_sh = T.alloc_shared((BLOCK_M, BLOCK_N), dtype)

                # gmem -> smem
                T.copy(A[by * BLOCK_M:(by + 1) * BLOCK_M,
                        bx * BLOCK_N:(bx + 1) * BLOCK_N], A_sh)
                T.copy(A[by * BLOCK_M:(by + 1) * BLOCK_M,
                        (bx + n_block_num) * BLOCK_N:(bx + n_block_num + 1) * BLOCK_N], B_sh)

                # silu_mul for each element
                for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                    xi = A_sh[i, j].astype(accum_dtype)      # 先转 fp32 求 sigmoid 更稳
                    sig = 1.0 / (1.0 + T.exp(-xi))           # sigmoid
                    C_sh[i, j] = (xi * sig * B_sh[i, j]).astype(dtype)

                # smem -> gmem
                T.copy(C_sh, C[by * BLOCK_M:(by + 1) * BLOCK_M,
                            bx * BLOCK_N:(bx + 1) * BLOCK_N])

        return silu_mul
    
class MicroSiluMul(BaseMicroKernel):
    def __init__(self, M, N, dtype=T.bfloat16, accum_dtype=T.float32):
        super().__init__()
        
        self.M = M
        self.N = N
        self.dtype = dtype
        self.accum_dtype = accum_dtype
        self.strategy = _SiluMulStrategy(M, N, dtype, accum_dtype)
        
    def get_source(self, kernel, selected_hparams):
        head_str = \
'''
namespace kernel {

template <typename T,
          int THREAD_NUM,
          int TILE_DIM_X, 
          int TILE_DIM_Y, 
          int TILE_DIM_Z,
          int M,
          int N,
          int I_STRIDE,
          int O_STRIDE>
__device__ __forceinline__ void silu_mul_kernel_<name_suffix>(const int bx, const int by, const int bz,
                                                   void const *input_ptr,
                                                   void *output_ptr,
                                                   int num_active_tokens) {
  static_assert(THREAD_NUM==<threads>);
  static_assert(TILE_DIM_X==<BLOCK_N>); static_assert(TILE_DIM_Y==<BLOCK_M>); static_assert(TILE_DIM_Z==<BLOCK_K>);
  static_assert(M==<M>); static_assert(N==<N>);
  
  const <dtype>* __restrict__ A = static_cast<const <dtype>*>(input_ptr);
  <dtype>* __restrict__ C = static_cast<<dtype>*>(output_ptr);
  
'''     
        BLOCK_M, BLOCK_N, threads = selected_hparams
        BLOCK_K = 1
        
        head_str = head_str.replace('<threads>', str(threads))
        head_str = head_str.replace('<BLOCK_M>', str(BLOCK_M))
        head_str = head_str.replace('<BLOCK_N>', str(BLOCK_N)) 
        head_str = head_str.replace('<BLOCK_K>', str(BLOCK_K)) 
        head_str = head_str.replace('<M>', str(self.M))
        head_str = head_str.replace('<N>', str(self.N)) 
        head_str = head_str.replace('<name_suffix>', str(self.M)+"_"+str(self.N))
        if self.dtype == T.bfloat16:
            dtype = "bfloat16_t"
        else:
            dtype = "float16_t"
        head_str = head_str.replace('<dtype>', str(dtype))
                
        origin_source = kernel.get_kernel_source()
        source = origin_source.replace("blockIdx.x", "bx")
        source = source.replace("blockIdx.y", "by")
        source = source.replace("blockIdx.z", "bz")
        source = self.replace_line(source, "extern \"C\" __global__", 1, head_str)
        source += "\n} // kernel"
        
        grid_dim, block_dim, dynamic_smem_buf, use_cooperative_groups = kernel.get_launch_info()
        self.grid_tile_info = f"grid_dim=({grid_dim['blockIdx.x']}, {grid_dim['blockIdx.y']}, {grid_dim['blockIdx.z']}), tile_dim=({BLOCK_N}, {BLOCK_M}, {BLOCK_K})"
        extra_attr = f"\n// Strategy: {self.strategy.name}"
        extra_attr += f"\n// selected_hparams: {selected_hparams}."
        extra_attr += f"\n// smem: {dynamic_smem_buf} bytes."
        extra_attr += f"\n// use_cooperative_groups: {use_cooperative_groups}."
        extra_attr += f"\n// " + self.grid_tile_info
        extra_attr += f"\n// block_dim=({block_dim['threadIdx.x']}, {block_dim['threadIdx.y']}, {block_dim['threadIdx.z']})."
        source += extra_attr
        
        return source
    
    def get_kernel(self, mode: HparamSelectMode):
        kernel, path = self.auto_get_kernel(self.get_source, self.strategy, mode)
        return kernel, path, self.grid_tile_info