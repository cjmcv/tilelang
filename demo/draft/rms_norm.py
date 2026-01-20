import torch
import tilelang
import tilelang.language as T
from tilelang.utils.profiler import do_bench
from common.pkt_util import TestUtil, TorchRef


@tilelang.jit(out_idx=[-1], pass_configs={"tl.disable_tma_lower": True})
def rms_norm_splitk(M, N, blk_m, blk_k, eps=1e-12, dtype="bfloat16", accum_dtype="float32"):
    
    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, blk_m), threads=128) as bx:
            A_shared = T.alloc_shared((blk_m, blk_k), dtype)
            A_local = T.alloc_fragment((blk_m, blk_k), accum_dtype)
            A_powsum = T.alloc_fragment((blk_m,), accum_dtype)

            num_k_step = T.ceildiv(N, blk_k)
            T.clear(A_local)
            for k in range(num_k_step):
                T.copy(A[bx * blk_m, k * blk_k], A_shared)
                for i, j in T.Parallel(blk_m, blk_k):
                    A_local[i, j] += A_shared[i, j] * A_shared[i, j]
            T.reduce_sum(A_local, A_powsum, dim=1)
            for i in T.Parallel(blk_m):
                A_powsum[i] = T.rsqrt(A_powsum[i] / N + eps)

            for k in range(num_k_step):
                # reverse, better cache hit rate
                T.copy(A[bx * blk_m, (num_k_step - 1 - k) * blk_k], A_shared)
                for i, j in T.Parallel(blk_m, blk_k):
                    A_shared[i, j] *= A_powsum[i]
                T.copy(A_shared, B[bx * blk_m, (num_k_step - 1 - k) * blk_k])

    return main


@tilelang.jit(out_idx=[-1], pass_configs={"tl.disable_tma_lower": True})
def rms_norm(M, N, blk_m, threads, eps=1e-12, dtype="bfloat16", accum_dtype="float32"):
    
    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((1, N), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, blk_m), threads=threads) as bx:
            A_shared = T.alloc_shared((blk_m, N), dtype)
            A_pow_local = T.alloc_fragment((blk_m, N), accum_dtype)
            A_local = T.alloc_fragment((blk_m, N), accum_dtype)
            A_powsum = T.alloc_fragment((blk_m,), accum_dtype)
            B_shared = T.alloc_shared((1, N), dtype)
            B_local = T.alloc_fragment((1, N), accum_dtype)
            
            T.copy(A[bx * blk_m : (bx + 1) * blk_m, :], A_shared)
            T.copy(B[0:1, :], B_shared)
            
            T.copy(A_shared, A_local)
            T.copy(B_shared, B_local)
            
            for i, j in T.Parallel(blk_m, N):
                A_pow_local[i, j] = A_local[i, j] * A_local[i, j]
            T.reduce_sum(A_pow_local, A_powsum, dim=1)
            for i in T.Parallel(blk_m):
                A_powsum[i] = T.rsqrt(A_powsum[i] / N + eps)
            for i, j in T.Parallel(blk_m, N):
                A_local[i, j] *= A_powsum[i] * B_local[0, j]
            T.copy(A_local, C[bx * blk_m : (bx + 1) * blk_m, :])

    return main


def ref_program(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-12) * w


if __name__ == "__main__":
    M, N, blk_m, blk_k = 1, 2560, 1, 512
    kernel = rms_norm(M, N, blk_m, threads=128)

    a = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(1, N, dtype=torch.bfloat16, device="cuda")
    c = kernel(a, b)
    print("c:")
    print(c)
    ref_c = TorchRef.rms_norm(a, b)
    print("ref_c:")
    print(ref_c)
    torch.testing.assert_close(c, ref_c, rtol=1e-1, atol=1e-1)
    
    # benchmark
    latency = do_bench(lambda: TorchRef.rms_norm(a, b), warmup=500, backend="cupti")
    print(f"torch Latency: {latency}ms")
    
    latency = do_bench(lambda: kernel(a, b), warmup=500, backend="cupti")
    print(f"tilelang Latency: {latency}ms")
