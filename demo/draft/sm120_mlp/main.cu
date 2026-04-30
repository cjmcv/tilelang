// main.cu - Demo for fused MLP: rms_norm + gemm + silu_mul + gemm_add
// Complete MLP flow: input -> rms_norm -> gemm1 -> silu_mul -> gemm2_add -> output

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

#include "rms_norm_tl_1_1024.cuh"
#include "linear_gemm_tl_1_6144_1024.cuh"
#include "silu_mul_tl_1_3072.cuh"
#include "linear_gemm_add_tl_1_1024_3072.cuh"

namespace {

#define CHECK_CU(result)                                                       \
  do {                                                                         \
    CUresult _result = (result);                                               \
    if (_result != CUDA_SUCCESS) {                                              \
      const char *errstr;                                                       \
      cuGetErrorString(_result, &errstr);                                      \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, errstr);\
      exit(1);                                                                  \
    }                                                                          \
  } while (0)

#define CHECK_RT(result)                                                        \
  do {                                                                         \
    cudaError_t _result = (result);                                             \
    if (_result != cudaSuccess) {                                               \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,        \
              cudaGetErrorName(_result));                                      \
      exit(1);                                                                  \
    }                                                                          \
  } while (0)

}  // anonymous namespace

// #define TEST_MEGA 1
// __global__ void __launch_bounds__(256, 1) mega_mlp(void const *rms_input_ptr, void const *rms_weight_ptr, void *rms_output_ptr,
//                                                    const CUtensorMap *gemm_A_desc,  const CUtensorMap *gemm_B_desc,  const CUtensorMap *gemm_C_desc,
//                                                    void const *silu_mul_input_ptr, void *silu_mul_output_ptr,
//                                                    const CUtensorMap *gemm_add_A_desc,  const CUtensorMap *gemm_add_B_desc,  const CUtensorMap *gemm_add_C_desc, const void* __restrict__ gemm_add_R_ptr,) {
//     kernel::rms_norm_kernel_1_1024<bfloat16_t, 256, 1, 1, 1, 1, 1024>(
//         blockIdx.x, blockIdx.y, blockIdx.z,
//         rms_input_ptr,
//         rms_weight_ptr,
//         rms_output_ptr,
//         1e-12f);
    
//     kernel::linear_gemm_tl_1_6144_1024<bfloat16_t, 256, 64, 16, 128, 1, 6144, 1024, 6144, 3, false>(
//         blockIdx.x, blockIdx.y, blockIdx.z,
//         gemm_A_desc,
//         gemm_B_desc,
//         nullptr,
//         gemm_C_desc,
//         1,
//         false/*residual*/);

//     kernel::silu_mul_kernel_1_3072<bfloat16_t, 256, 64, 32, 1, 1, 3072, 6144, 3072>(
//         blockIdx.x, blockIdx.y, blockIdx.z,
//         silu_mul_input_ptr,
//         silu_mul_output_ptr,
//         1);

//     kernel::linear_gemm_add_tl_1_1024_3072<bfloat16_t, 256, 64, 16, 128, 1, 1024, 3072, 1024, 3, true>(
//       blockIdx.x, blockIdx.y, blockIdx.z,
//       gemm_add_A_desc,
//       gemm_add_B_desc,
//       gemm_add_R_ptr,
//       gemm_add_C_desc,
//       1,
//       false/*residual*/);
// }

int main(int argc, char **argv) {
    // MLP dimensions
    constexpr int M = 1;
    constexpr int HIDDEN_SIZE = 1024;
    constexpr int INTERMEDIATE_SIZE = 3072;
    constexpr int GATED_UP_SIZE = INTERMEDIATE_SIZE * 2;  // 6144

    // Kernel configurations
    constexpr int RMS_NORM_BLOCK = 256;
    constexpr size_t RMS_NORM_SMEM_SIZE = 4096;
    constexpr int GEMM1_GRID_X = 96;
    constexpr int GEMM1_THREAD_NUM = 256;
    constexpr size_t GEMM1_SMEM_SIZE = 61440;
    constexpr int SILU_MUL_GRID_X = 48;
    constexpr int SILU_MUL_THREAD_NUM = 256;
    constexpr size_t SILU_MUL_SMEM_SIZE = 12288;
    constexpr int GEMM2_GRID_X = 16;
    constexpr int GEMM2_THREAD_NUM = 256;
    constexpr size_t GEMM2_SMEM_SIZE = 20480;

    printf("=== Fused MLP Demo ===\n");
    printf("MLP dimensions: hidden_size=%d, intermediate_size=%d, gated_up_size=%d\n",
           HIDDEN_SIZE, INTERMEDIATE_SIZE, GATED_UP_SIZE);
    printf("\nMLP Flow:\n");
    printf("  1. rms_norm:     input[%d,%d] -> rms_out[%d,%d]\n", M, HIDDEN_SIZE, M, HIDDEN_SIZE);
    printf("  2. gemm1:        rms_out[%d,%d] x w_gatedup[%d,%d] -> mlp_mid[%d,%d]\n",
           M, HIDDEN_SIZE, GATED_UP_SIZE, HIDDEN_SIZE, M, GATED_UP_SIZE);
    printf("  3. silu_mul:     mlp_mid[%d,%d] -> silu_mul_out[%d,%d]\n", M, GATED_UP_SIZE, M, INTERMEDIATE_SIZE);
    printf("  4. gemm2_add:    silu_mul_out[%d,%d] x w_down_proj[%d,%d] + residual -> mlp_out[%d,%d]\n",
           M, INTERMEDIATE_SIZE, HIDDEN_SIZE, INTERMEDIATE_SIZE, M, HIDDEN_SIZE);

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
    bfloat16_t *d_input = nullptr;       // [1, 1024] - original input, also used as residual
    bfloat16_t *d_rms_out = nullptr;    // [1, 1024] - after rms_norm
    bfloat16_t *d_mlp_mid = nullptr;    // [1, 6144] - after first gemm (gated up)
    bfloat16_t *d_silu_mul_out = nullptr; // [1, 3072] - after silu_mul
    bfloat16_t *d_mlp_out = nullptr;    // [1, 1024] - final output

    CHECK_RT(cudaMalloc(&d_input, M * HIDDEN_SIZE * sizeof(bfloat16_t)));
    CHECK_RT(cudaMalloc(&d_rms_out, M * HIDDEN_SIZE * sizeof(bfloat16_t)));
    CHECK_RT(cudaMalloc(&d_mlp_mid, M * GATED_UP_SIZE * sizeof(bfloat16_t)));
    CHECK_RT(cudaMalloc(&d_silu_mul_out, M * INTERMEDIATE_SIZE * sizeof(bfloat16_t)));
    CHECK_RT(cudaMalloc(&d_mlp_out, M * HIDDEN_SIZE * sizeof(bfloat16_t)));

    // Allocate device memory for weights
    bfloat16_t *d_w_rms_norm = nullptr;   // [1, 1024]
    bfloat16_t *d_w_gatedup = nullptr;    // [6144, 1024]
    bfloat16_t *d_w_down_proj = nullptr;  // [1024, 3072]

    CHECK_RT(cudaMalloc(&d_w_rms_norm, M * HIDDEN_SIZE * sizeof(bfloat16_t)));
    CHECK_RT(cudaMalloc(&d_w_gatedup, GATED_UP_SIZE * HIDDEN_SIZE * sizeof(bfloat16_t)));
    CHECK_RT(cudaMalloc(&d_w_down_proj, HIDDEN_SIZE * INTERMEDIATE_SIZE * sizeof(bfloat16_t)));

    // Initialize input data
    std::vector<bfloat16_t> h_input(M * HIDDEN_SIZE);
    std::vector<bfloat16_t> h_w_rms_norm(M * HIDDEN_SIZE);
    std::vector<bfloat16_t> h_w_gatedup(GATED_UP_SIZE * HIDDEN_SIZE);
    std::vector<bfloat16_t> h_w_down_proj(HIDDEN_SIZE * INTERMEDIATE_SIZE);

    for (int i = 0; i < M * HIDDEN_SIZE; i++) {
        h_input[i] = bfloat16_t((float(i % HIDDEN_SIZE) / HIDDEN_SIZE) * 0.01f + 0.001f);
        h_w_rms_norm[i] = bfloat16_t(1.0f);
    }
    for (int i = 0; i < GATED_UP_SIZE * HIDDEN_SIZE; i++) {
        h_w_gatedup[i] = bfloat16_t((float(i % HIDDEN_SIZE) / HIDDEN_SIZE) * 0.01f + 0.001f);
    }
    for (int i = 0; i < HIDDEN_SIZE * INTERMEDIATE_SIZE; i++) {
        h_w_down_proj[i] = bfloat16_t((float(i % INTERMEDIATE_SIZE) / INTERMEDIATE_SIZE) * 0.01f + 0.001f);
    }

    // Copy data to device
    CHECK_RT(cudaMemcpy(d_input, h_input.data(), M * HIDDEN_SIZE * sizeof(bfloat16_t), cudaMemcpyHostToDevice));
    CHECK_RT(cudaMemcpy(d_w_rms_norm, h_w_rms_norm.data(), M * HIDDEN_SIZE * sizeof(bfloat16_t), cudaMemcpyHostToDevice));
    CHECK_RT(cudaMemcpy(d_w_gatedup, h_w_gatedup.data(), GATED_UP_SIZE * HIDDEN_SIZE * sizeof(bfloat16_t), cudaMemcpyHostToDevice));
    CHECK_RT(cudaMemcpy(d_w_down_proj, h_w_down_proj.data(), HIDDEN_SIZE * INTERMEDIATE_SIZE * sizeof(bfloat16_t), cudaMemcpyHostToDevice));

    // Create TMA descriptors for gemm kernels
    CUtensorMap *A1_desc, *B1_desc, *C1_desc;
    cudaMalloc(&A1_desc, sizeof(CUtensorMap));
    cudaMalloc(&B1_desc, sizeof(CUtensorMap));
    cudaMalloc(&C1_desc, sizeof(CUtensorMap));

    CUtensorMap *A2_desc, *B2_desc, *C2_desc;
    cudaMalloc(&A2_desc, sizeof(CUtensorMap));
    cudaMalloc(&B2_desc, sizeof(CUtensorMap));
    cudaMalloc(&C2_desc, sizeof(CUtensorMap));

    // Create TMA descriptors for gemm1 (rms_out x w_gatedup -> mlp_mid)
    // A: rms_out [1, 1024], B: w_gatedup [6144, 1024], C: mlp_mid [1, 6144]
    printf("\nCreating TMA descriptors for gemm1...\n");
    int ret = create_linear_gemm_tl_1_6144_1024(d_rms_out, d_w_gatedup, d_mlp_mid, A1_desc, B1_desc, C1_desc, true);
    if (ret != 0) {
        printf("ERROR: create_linear_gemm_tl_1_6144_1024 failed with code %d\n", ret);
        return 1;
    }

    // Create TMA descriptors for gemm2 (silu_mul_out x w_down_proj + residual -> mlp_out)
    // A: silu_mul_out [1, 3072], B: w_down_proj [1024, 3072], C: mlp_out [1, 1024]
    printf("Creating TMA descriptors for gemm2...\n");
    ret = create_linear_gemm_add_tl_1_1024_3072(d_silu_mul_out, d_w_down_proj, d_input, d_mlp_out, A2_desc, B2_desc, C2_desc, true);
    if (ret != 0) {
        printf("ERROR: create_linear_gemm_add_tl_1_1024_3072 failed with code %d\n", ret);
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

    CHECK_RT(cudaFuncSetAttribute(
        linear_gemm_add_tl,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        GEMM2_SMEM_SIZE));

    cudaEvent_t start, stop;
    CHECK_RT(cudaEventCreate(&start));
    CHECK_RT(cudaEventCreate(&stop));

    printf("\n=== Running MLP kernels ===\n");

    // Warmup runs
    printf("\nWarmup: running 10 iterations...\n");
    for (int i = 0; i < 10; i++) {
        rms_norm_tl<<<dim3(1, 1, 1), dim3(RMS_NORM_BLOCK, 1, 1), RMS_NORM_SMEM_SIZE, stream>>>(
            d_input, d_w_rms_norm, d_rms_out);
        linear_gemm_tl<<<dim3(GEMM1_GRID_X, 1, 1), dim3(GEMM1_THREAD_NUM, 1, 1), GEMM1_SMEM_SIZE, stream>>>(
            A1_desc, B1_desc, C1_desc);
        silu_mul_tl<<<dim3(SILU_MUL_GRID_X, 1, 1), dim3(SILU_MUL_THREAD_NUM, 1, 1), SILU_MUL_SMEM_SIZE, stream>>>(
            d_mlp_mid, d_silu_mul_out);
        linear_gemm_add_tl<<<dim3(GEMM2_GRID_X, 1, 1), dim3(GEMM2_THREAD_NUM, 1, 1), GEMM2_SMEM_SIZE, stream>>>(
            A2_desc, B2_desc, C2_desc, (void *)d_input);
    }
    CHECK_RT(cudaStreamSynchronize(stream));
    printf("Warmup done!\n");

// ========== Step 1: rms_norm ==========
    printf("\n[Step 1] rms_norm: input -> rms_out\n");
    CHECK_RT(cudaFuncSetAttribute(
        rms_norm_tl,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        RMS_NORM_SMEM_SIZE));
    CHECK_RT(cudaEventRecord(start, stream));
rms_norm_tl<<<dim3(1, 1, 1), dim3(RMS_NORM_BLOCK, 1, 1), RMS_NORM_SMEM_SIZE, stream>>>(
        d_input, d_w_rms_norm, d_rms_out);
    CHECK_RT(cudaEventRecord(stop, stream));
    CHECK_RT(cudaStreamSynchronize(stream));
    float rms_time = 0.0f;
    CHECK_RT(cudaEventElapsedTime(&rms_time, start, stop));
    printf("  rms_norm completed! Time: %.3f ms\n", rms_time);

    // ========== Step 2: gemm1 (gated up) ==========
    printf("\n[Step 2] gemm1 (gated up): rms_out x w_gatedup -> mlp_mid\n");
    CHECK_RT(cudaEventRecord(start, stream));
    linear_gemm_tl<<<dim3(GEMM1_GRID_X, 1, 1), dim3(GEMM1_THREAD_NUM, 1, 1), GEMM1_SMEM_SIZE, stream>>>(
        A1_desc, B1_desc, C1_desc);
    CHECK_RT(cudaEventRecord(stop, stream));
    CHECK_RT(cudaStreamSynchronize(stream));
    float gemm1_time = 0.0f;
    CHECK_RT(cudaEventElapsedTime(&gemm1_time, start, stop));
    printf("  gemm1 completed! Time: %.3f ms\n", gemm1_time);

    // ========== Step 3: silu_mul ==========
    printf("\n[Step 3] silu_mul: mlp_mid -> silu_mul_out\n");
    CHECK_RT(cudaEventRecord(start, stream));
    silu_mul_tl<<<dim3(SILU_MUL_GRID_X, 1, 1), dim3(SILU_MUL_THREAD_NUM, 1, 1), SILU_MUL_SMEM_SIZE, stream>>>(
        d_mlp_mid, d_silu_mul_out);
    CHECK_RT(cudaEventRecord(stop, stream));
    CHECK_RT(cudaStreamSynchronize(stream));
    float silu_time = 0.0f;
    CHECK_RT(cudaEventElapsedTime(&silu_time, start, stop));
    printf("  silu_mul completed! Time: %.3f ms\n", silu_time);

    // ========== Step 4: gemm2 with residual ==========
    printf("\n[Step 4] gemm2 (down_proj + residual): silu_mul_out x w_down_proj + residual -> mlp_out\n");
    CHECK_RT(cudaEventRecord(start, stream));
    linear_gemm_add_tl<<<dim3(GEMM2_GRID_X, 1, 1), dim3(GEMM2_THREAD_NUM, 1, 1), GEMM2_SMEM_SIZE, stream>>>(
        A2_desc, B2_desc, C2_desc, (void *)d_input);
    CHECK_RT(cudaEventRecord(stop, stream));
    CHECK_RT(cudaStreamSynchronize(stream));
    float gemm2_time = 0.0f;
    CHECK_RT(cudaEventElapsedTime(&gemm2_time, start, stop));
    printf("  gemm2 completed! Time: %.3f ms\n", gemm2_time);

    float total_time = rms_time + gemm1_time + silu_time + gemm2_time;
    printf("\n=== MLP Kernel Summary ===\n");
    printf("  rms_norm:    %.3f ms\n", rms_time);
    printf("  gemm1:       %.3f ms\n", gemm1_time);
    printf("  silu_mul:    %.3f ms\n", silu_time);
    printf("  gemm2_add:   %.3f ms\n", gemm2_time);
    printf("  Total:       %.3f ms\n", total_time);

    // Copy result back for verification
    std::vector<bfloat16_t> h_mlp_out(M * HIDDEN_SIZE);
    CHECK_RT(cudaMemcpy(h_mlp_out.data(), d_mlp_out, M * HIDDEN_SIZE * sizeof(bfloat16_t), cudaMemcpyDeviceToHost));

    // ========== Reference verification ==========
    printf("\n=== Verifying results (comparing with PyTorch reference) ===\n");

    // Compute reference: MLP = rms_norm(x) @ w_gatedup -> silu_mul -> @ w_down_proj + x
    std::vector<float> h_input_float(M * HIDDEN_SIZE);
    std::vector<float> h_rms_out_float(M * HIDDEN_SIZE);
    std::vector<float> h_mlp_mid_float(M * GATED_UP_SIZE);
    std::vector<float> h_silu_mul_out_float(M * INTERMEDIATE_SIZE);
    std::vector<float> h_ref_output(M * HIDDEN_SIZE);

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
        h_mlp_mid_float[j] = sum;
    }

    // Reference: silu_mul (silu(x) * gate)
    for (int i = 0; i < INTERMEDIATE_SIZE; i++) {
        float x_val = h_mlp_mid_float[i];
        float gate_val = h_mlp_mid_float[i + INTERMEDIATE_SIZE];
        float sigmoid_x = 1.0f / (1.0f + expf(-x_val));
        h_silu_mul_out_float[i] = x_val * sigmoid_x * gate_val;
    }

    // Reference: second gemm with residual
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        float sum = 0.0f;
        for (int k = 0; k < INTERMEDIATE_SIZE; k++) {
            sum += h_silu_mul_out_float[k] * (float)h_w_down_proj[j * INTERMEDIATE_SIZE + k];
        }
        h_ref_output[j] = sum + h_input_float[j];  // add residual
    }

    // Compare
    double max_abs_diff = 0.0;
    int max_diff_idx = 0;
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        float diff = fabs((float)h_mlp_out[i] - h_ref_output[i]);
        if (diff > max_abs_diff) {
            max_abs_diff = diff;
            max_diff_idx = i;
        }
    }

    printf("Max absolute diff: %.6f (at index %d)\n", max_abs_diff, max_diff_idx);
    printf("  Computed: %.6f, Reference: %.6f\n", (float)h_mlp_out[max_diff_idx], h_ref_output[max_diff_idx]);

    // Show first 10 results
    printf("\nFirst 10 results:\n");
    for (int i = 0; i < 10 && i < HIDDEN_SIZE; i++) {
        printf("  mlp_out[%d]: kernel=%.6f ref=%.6f diff=%.6f\n",
               i, (float)h_mlp_out[i], h_ref_output[i], fabs((float)h_mlp_out[i] - h_ref_output[i]));
    }

    if (max_abs_diff < 0.1f) {
        printf("\nPASS: Results match! (max_diff=%.6f)\n", max_abs_diff);
    } else {
        printf("\nWARNING: Results differ more than expected (max_diff=%.6f)\n", max_abs_diff);
    }

    // Cleanup
    CHECK_RT(cudaFree(d_input));
    CHECK_RT(cudaFree(d_rms_out));
    CHECK_RT(cudaFree(d_mlp_mid));
    CHECK_RT(cudaFree(d_silu_mul_out));
    CHECK_RT(cudaFree(d_mlp_out));
    CHECK_RT(cudaFree(d_w_rms_norm));
    CHECK_RT(cudaFree(d_w_gatedup));
    CHECK_RT(cudaFree(d_w_down_proj));
    CHECK_RT(cudaFree(A1_desc));
    CHECK_RT(cudaFree(B1_desc));
    CHECK_RT(cudaFree(C1_desc));
    CHECK_RT(cudaFree(A2_desc));
    CHECK_RT(cudaFree(B2_desc));
    CHECK_RT(cudaFree(C2_desc));
    CHECK_RT(cudaStreamDestroy(stream));

    printf("\nDone!\n");
    return 0;
}