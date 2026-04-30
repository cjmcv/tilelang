#include "linear_gemm_tl_1_6144_1024_split_merge.cuh"

namespace {

#define CHECK_CU(result)                                                       \
  do {                                                                         \
    CUresult _result = (result);                                               \
    if (_result != CUDA_SUCCESS) {                                              \
      const char *errstr;                                                       \
      cuGetErrorString(_result, &errstr);                                      \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, errstr);\
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define CHECK_RT(result)                                                        \
  do {                                                                         \
    cudaError_t _result = (result);                                             \
    if (_result != cudaSuccess) {                                               \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,        \
              cudaGetErrorName(_result));                                      \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

}  // anonymous namespace


int main(int argc, char **argv) {
    // MLP dimensions
    constexpr int M = 1;
    constexpr int HIDDEN_SIZE = 1024;
    constexpr int GATED_UP_SIZE = 6144;  // For linear_gemm_tl_1_6144_1024

    // Kernel configurations
    constexpr int RMS_NORM_BLOCK = 256;
    constexpr size_t RMS_NORM_SMEM_SIZE = 4096;
    constexpr int GEMM1_GRID_X = 96;
    constexpr int GEMM1_THREAD_NUM = 256;
    constexpr size_t GEMM1_SMEM_SIZE = 61440;

    printf("=== Fused MLP Demo (rms_norm + gemm) ===\n");
    printf("MLP dimensions: hidden_size=%d, gated_up_size=%d\n",
           HIDDEN_SIZE, GATED_UP_SIZE);
    printf("\nMLP Flow:\n");
    printf("  1. rms_norm:     input[%d,%d] -> rms_out[%d,%d]\n", M, HIDDEN_SIZE, M, HIDDEN_SIZE);
    printf("  2. gemm1:        rms_out[%d,%d] x w_gatedup[%d,%d] -> mlp_mid[%d,%d]\n",
           M, HIDDEN_SIZE, GATED_UP_SIZE, HIDDEN_SIZE, M, GATED_UP_SIZE);

    // Initialize runtime API
    CHECK_RT(cudaSetDevice(0));
    cudaDeviceProp prop;
    CHECK_RT(cudaGetDeviceProperties(&prop, 0));
    printf("\nDevice: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    if (prop.major < 9) {
        printf("ERROR: This kernel requires Hopper (sm_90a)\n");
        return 1;
    }

    // Allocate device memory for inputs/outputs
    bfloat16_t *d_input = nullptr;       // [1, 1024] - original input
    bfloat16_t *d_rms_out = nullptr;    // [1, 1024] - after rms_norm
    bfloat16_t *d_mlp_mid = nullptr;    // [1, 6144] - after first gemm (gated up)

    CHECK_RT(cudaMalloc(&d_input, M * HIDDEN_SIZE * sizeof(bfloat16_t)));
    CHECK_RT(cudaMalloc(&d_rms_out, M * HIDDEN_SIZE * sizeof(bfloat16_t)));
    CHECK_RT(cudaMalloc(&d_mlp_mid, M * GATED_UP_SIZE * sizeof(bfloat16_t)));

    // Allocate device memory for weights
    bfloat16_t *d_w_rms_norm = nullptr;   // [1, 1024]
    bfloat16_t *d_w_gatedup = nullptr;    // [6144, 1024]

    CHECK_RT(cudaMalloc(&d_w_rms_norm, M * HIDDEN_SIZE * sizeof(bfloat16_t)));
    CHECK_RT(cudaMalloc(&d_w_gatedup, GATED_UP_SIZE * HIDDEN_SIZE * sizeof(bfloat16_t)));

    // Initialize input data
    std::vector<bfloat16_t> h_input(M * HIDDEN_SIZE);
    std::vector<bfloat16_t> h_w_rms_norm(M * HIDDEN_SIZE);
    std::vector<bfloat16_t> h_w_gatedup(GATED_UP_SIZE * HIDDEN_SIZE);

    for (int i = 0; i < M * HIDDEN_SIZE; i++) {
        h_input[i] = bfloat16_t((float(i % HIDDEN_SIZE) / HIDDEN_SIZE) * 0.01f + 0.001f);
        h_w_rms_norm[i] = bfloat16_t(1.0f);
    }
    for (int i = 0; i < GATED_UP_SIZE * HIDDEN_SIZE; i++) {
        h_w_gatedup[i] = bfloat16_t((float(i % HIDDEN_SIZE) / HIDDEN_SIZE) * 0.01f + 0.001f);
    }

    // Copy data to device
    CHECK_RT(cudaMemcpy(d_input, h_input.data(), M * HIDDEN_SIZE * sizeof(bfloat16_t), cudaMemcpyHostToDevice));
    CHECK_RT(cudaMemcpy(d_w_rms_norm, h_w_rms_norm.data(), M * HIDDEN_SIZE * sizeof(bfloat16_t), cudaMemcpyHostToDevice));
    CHECK_RT(cudaMemcpy(d_w_gatedup, h_w_gatedup.data(), GATED_UP_SIZE * HIDDEN_SIZE * sizeof(bfloat16_t), cudaMemcpyHostToDevice));

    // Create TMA descriptors for gemm kernels
    CUtensorMap *A_desc, *B_desc, *C_desc;
    cudaMalloc(&A_desc, sizeof(CUtensorMap));
    cudaMalloc(&B_desc, sizeof(CUtensorMap));
    cudaMalloc(&C_desc, sizeof(CUtensorMap));

    // Create TMA descriptors for gemm1 (rms_out x w_gatedup -> mlp_mid)
    // A: rms_out [1, 1024], B: w_gatedup [6144, 1024], C: mlp_mid [1, 6144]
    // Note: C pointer should match A_desc - the kernel writes output via A_desc pointer
    printf("\nCreating TMA descriptors for gemm1...\n");
    int ret = create_linear_gemm_tl_1_6144_1024(d_rms_out, d_w_gatedup, d_rms_out, A_desc, B_desc, C_desc, true);
    if (ret != 0) {
        printf("ERROR: create_linear_gemm_tl_1_6144_1024 failed with code %d\n", ret);
        return 1;
    }

    // Create stream
    cudaStream_t stream;
    CHECK_RT(cudaStreamCreate(&stream));

    // Set max dynamic shared memory for kernels
    CHECK_RT(cudaFuncSetAttribute(
        linear_gemm_tl,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        GEMM1_SMEM_SIZE));

    cudaEvent_t start, stop;
    CHECK_RT(cudaEventCreate(&start));
    CHECK_RT(cudaEventCreate(&stop));

    printf("\n=== Running MLP kernels ===\n");

    // Warmup runs
    printf("\nWarmup: running 10 iterations...\n");
    for (int i = 0; i < 10; i++) {
        linear_gemm_tl<<<dim3(GEMM1_GRID_X, 1, 1), dim3(GEMM1_THREAD_NUM, 1, 1), GEMM1_SMEM_SIZE, stream>>>(
            d_input, d_w_rms_norm, d_rms_out, A_desc, B_desc, C_desc);
    }
    CHECK_RT(cudaStreamSynchronize(stream));
    printf("Warmup done!\n");

    // ========== Step 1+2: fused rms_norm + gemm1 ==========
    printf("\n[Step 1+2] fused rms_norm + gemm1: input -> rms_out -> mlp_mid\n");
    CHECK_RT(cudaEventRecord(start, stream));
    linear_gemm_tl<<<dim3(GEMM1_GRID_X, 1, 1), dim3(GEMM1_THREAD_NUM, 1, 1), GEMM1_SMEM_SIZE, stream>>>(
        d_input, d_w_rms_norm, d_rms_out, A_desc, B_desc, C_desc);
    CHECK_RT(cudaEventRecord(stop, stream));
    CHECK_RT(cudaStreamSynchronize(stream));
    float total_time = 0.0f;
    CHECK_RT(cudaEventElapsedTime(&total_time, start, stop));
    printf("  fused kernel completed! Time: %.3f ms\n", total_time);

    // Copy result back for verification
    std::vector<bfloat16_t> h_rms_out(M * HIDDEN_SIZE);
    std::vector<bfloat16_t> h_mlp_mid(M * GATED_UP_SIZE);
    CHECK_RT(cudaMemcpy(h_rms_out.data(), d_rms_out, M * HIDDEN_SIZE * sizeof(bfloat16_t), cudaMemcpyDeviceToHost));
    CHECK_RT(cudaMemcpy(h_mlp_mid.data(), d_mlp_mid, M * GATED_UP_SIZE * sizeof(bfloat16_t), cudaMemcpyDeviceToHost));

    // ========== Reference verification ==========
    printf("\n=== Verifying results (comparing with PyTorch reference) ===\n");

    // Compute reference: MLP = rms_norm(x) @ w_gatedup
    std::vector<float> h_input_float(M * HIDDEN_SIZE);
    std::vector<float> h_rms_out_float(M * HIDDEN_SIZE);
    std::vector<float> h_ref_mlp_mid(M * GATED_UP_SIZE);

    for (int i = 0; i < M * HIDDEN_SIZE; i++) {
        h_input_float[i] = (float)h_input[i];
    }

    // Reference: rms_norm
    float sum_sq = 0.0f;
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        sum_sq += h_input_float[i] * h_input_float[i];
    }
    float rms_norm_factor = 1.0f / sqrtf(sum_sq / HIDDEN_SIZE + 1e-12f);
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        h_rms_out_float[i] = h_input_float[i] * rms_norm_factor * (float)h_w_rms_norm[i];
    }

    // Reference: first gemm (gated up)
    for (int j = 0; j < GATED_UP_SIZE; j++) {
        float sum = 0.0f;
        for (int k = 0; k < HIDDEN_SIZE; k++) {
            sum += h_rms_out_float[k] * (float)h_w_gatedup[j * HIDDEN_SIZE + k];
        }
        h_ref_mlp_mid[j] = sum;
    }

    // Compare rms_norm output
    double max_abs_diff_rms = 0.0;
    int max_diff_idx_rms = 0;
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        float diff = fabs((float)h_rms_out[i] - h_rms_out_float[i]);
        if (diff > max_abs_diff_rms) {
            max_abs_diff_rms = diff;
            max_diff_idx_rms = i;
        }
    }
    printf("rms_norm output - Max absolute diff: %.6f (at index %d)\n", max_abs_diff_rms, max_diff_idx_rms);

    // Compare gemm1 output
    double max_abs_diff_gemm = 0.0;
    int max_diff_idx_gemm = 0;
    for (int i = 0; i < GATED_UP_SIZE; i++) {
        float diff = fabs((float)h_mlp_mid[i] - h_ref_mlp_mid[i]);
        if (diff > max_abs_diff_gemm) {
            max_abs_diff_gemm = diff;
            max_diff_idx_gemm = i;
        }
    }
    printf("gemm1 output - Max absolute diff: %.6f (at index %d)\n", max_abs_diff_gemm, max_diff_idx_gemm);

    if (max_abs_diff_rms < 0.1f && max_abs_diff_gemm < 0.1f) {
        printf("\nPASS: Results match!\n");
    } else {
        printf("\nWARNING: Results differ more than expected\n");
    }

    // Cleanup
    CHECK_RT(cudaFree(d_input));
    CHECK_RT(cudaFree(d_rms_out));
    CHECK_RT(cudaFree(d_mlp_mid));
    CHECK_RT(cudaFree(d_w_rms_norm));
    CHECK_RT(cudaFree(d_w_gatedup));
    CHECK_RT(cudaFree(A_desc));
    CHECK_RT(cudaFree(B_desc));
    CHECK_RT(cudaFree(C_desc));
    CHECK_RT(cudaStreamDestroy(stream));

    printf("\nDone!\n");
    return 0;
}