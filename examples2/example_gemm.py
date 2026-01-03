import tilelang
import tilelang.language as T
import util

@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, block_K, dtype=T.bfloat16, accum_dtype=T.float32):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm


def main():
    kernel = matmul(1024, 1024, 1024, 128, 128, 32)

    import torch

    a = torch.randn((1024, 1024), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((1024, 1024), dtype=torch.bfloat16, device="cuda")

    c = kernel(a, b)

    ref_c = a @ b

    print("c:")
    print(c)
    print("ref_c:")
    print(ref_c)

    # torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    # print("All check passed.")

    # Get CUDA Source
    util.save_to_file(kernel.get_kernel_source(), "./gemm.cu")
    # util.save_to_file(kernel.get_host_source(), "./gemm.cpp")
    kernel.export_sources(kernel_path="./gen/gemm.cu", host_path="./gen/gemm.cpp")
    
    # benchmark
    profiler = kernel.get_profiler()
    latency = profiler.do_bench(backend="cupti")
    
    print("0:", kernel.prim_func.attrs)
    print("1:", kernel.adapter.params)
    print("2:", kernel.adapter.func)
    print("3:", kernel.config)
    # print("4:", kernel.prim_func)
    print("5:", kernel.prim_func.body)
    
    smem_bytes = util.get_smem_bytes(kernel.prim_func)
    print("shared memory =", smem_bytes, "B")

    latency = profiler.do_bench()
    print(f"tilelang Latency: {latency}ms")


if __name__ == "__main__":
    main()

# TODO: 套壳与mpk兼容，使一键生成mpk能使用的 linear_gemv_cutlass.cuh。
#       main.cu同时支持 测试整kernel / 测试micro kernel