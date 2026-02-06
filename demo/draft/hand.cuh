// q[batch, num_heads, seqlen_q, dim]
// kv[batch, num_heads, seqlen_kv, dim]
// mask*
// out[batch, num_heads, seqlen_q, dim]
// O=softmax(QK/sqrt(d))V
// QK = [seqlen_q, dim] * [seqlen_kv, dim] = [seqlen_q, seqlen_kv]
// PV = [seqlen_q, seqlen_kv] * [seqlen_kv, dim] = [seqlen_q, seqlen_kv] 

// let seqlen_q == seqlen_kv
// layout: [batch, head], threads: 128
__global__ void flashatten(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
                           int seqlen, int dim, float * __restrict__ out) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tile_size = seqlen * dim;
    extern float smem[];
    float *sQ = smem;
    float *sK = smem + tile_size;
    float *sV = smem + tile_size*2;
    float *sO = smem + tile_size*3;

    // float *l;
    // float *m;
    for () {
        // load kv smem
        for () {
            // load q smem

            // gemm(q,k)

            // softmax

            // gemm(pv)

            // update out
        }
    }
}

// A[m,k] * B[k,n] = C[m,n]
__global__ void gemm(const float* __restrict__ A, const float* __restrict__ B, const float* __restrict__ C, 
                     int M, int N, int K) {
    /*
    for(int i=0; i<M; i++) {
        for(int j=0; j<N; j++) {
            for(int k=0; k<K; k++) {
                C[i*N+j] += A[i*K+k] * B[k*N+j] 
            }
        }
    }
    */

}