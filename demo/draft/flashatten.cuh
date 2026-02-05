

// block: (batch, num_heads), threads: tile_q 
// 参数：N:  是seq_len，QKV需一样长，适用于self-attention的训练阶段，不适用于带kvcache的推理场景或cross-attention
//         (交叉注意力机制，q和kv分别来自不同seq， 用于decode-encoder模式， 常见llm模型如qwen3是Decoder-only)
//      d:  是head_dim
//      num_tile_q/tile_q: 将Q按行分成 num_tile_q 个块，每块 tile_q 行
//      num_tile_kv/tile_kv:  将KV按列分成 num_tile_kv 个块，每块 tile_kv 列
//      softmax_scale: 使用在 QK^T 之后，求max之前，sum *= softmax_scale，作为入参是为了灵活，而不是固定为1/sqrt(d)
//      l: log-sum-exp 或 sum, 指当前行的指数和（用于归一化）
//      m: max, 当前行的最大值（用于数值稳定性）
// 
// qkv的维度都是 (batch, num_heads, seq_len, head_dim)BHND布局同torch（H是head，N是seq_len,训练常用布局），对计算访存更友好。
//              sglang/vllm等需要seq_len放在num_heads前面，是BNHD布局，推理常用,因为追加token时，HD内存连续。
//              每个分块需要取ND与ND相乘，在BHND布局，ND内存连续，对计算访存友好。而BNHD布局，ND内存不连续，需要跨行才能取出一个block的数据。
// QK计算：(batch*num_heads(H), seq_len(N), head_dim(D)) * (batch, num_heads(H), seq_len(N), head_dim(D)) = (batch, num_heads, seq_len(N), seq_len(N))

// 外层循环遍历K/V块（j循环），内层循环遍历Q块（i循环）
// 每对(Q块, KV块)计算一次注意力，采用在线softmax算法增量更新
__global__
void forward_kernel(const float* Q, const float* K, const float* V, const int N, const int d,
                    const int num_tile_kv, const int num_tile_q, const int tile_kv, const int tile_q, 
                    const float softmax_scale, float* l, float *m, float* O) {
    int tx = threadIdx.x;
    int bx = blockIdx.x; int by = blockIdx.y;  // batch and head index

    //   d        d
    // N    x   N    = N x N
    //    d        d
    // n0   x   na   = n0xna  n0xnb
    // n1       nb     n1xna  n1xnb
    // n0/n1对应q的一个block内的小分块，大小为tile_q, tile_q*num_tile_q=N; na/nb对应kv的小分块，大小为tile_kv, tile_kv*num_tile_kv=N
    // 全局内存的偏移量，Offset into Q,K,V,O,l,m - different for each batch and head
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);  // gridDim.y = num_heads， bx对应batch维度， by对应head维度，切分后一个block对应一份(N * d)
    int lm_offset = (bx * gridDim.y * N) + (by * N);           // offset for l and m, 一个block对应一份N，QK计算后得到(N,N)，没有d维度. 每行都有一对l/m值, 正好对应每一个线程（下面都用tx充当行号）

    // N = num_tile_q × tile_q = num_tile_kv × tile_kv
    // 一个block对应一份(N*d)，一份(N*d)里会继续划分(tile_kv*d)的tile，
    // Define SRAM for Q,K,V,S
    extern __shared__ float sram[];
    int tile_size = tile_kv * d;  // size of Qi, Kj, Vj
    float* Qi = sram;
    float* Kj = &sram[tile_size];
    float* Vj = &sram[tile_size * 2];
    float* S = &sram[tile_size * 3]; // smem子tile的Q*K的输出矩阵: Q[tile_q, d] * K[tile_kv, d] = S[tile_q * tile_kv]

    // 遍历num_tile_kv，即遍历整份N的所有(tile_kv*d)的tile
    for (int j = 0; j < num_tile_kv; j++) {

        // 加载(tile_kv*d)的KV数据到smem，一个线程负责整行d个数据，tx充当行号，tx数量等于tile_kv
        // Load Kj, Vj to SRAM
        for (int x = 0; x < d; x++) {
            Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];
            Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];
        }
        __syncthreads();  // such that the inner loop can use the correct Kj, Vj

        // 一份kv块，与所有q块进行计算
        for (int i = 0; i < num_tile_q; i++)  {

            // 加载(tile_kv*d)的Q数据到smem，这里tile_kv应和tile_q大小一致，因为Qi[(tx * d) + x] 与 Kj[(tx * d) + x] 都在同样位置使用tx。
            // Load Qi to SRAM, l and m to registers
            for (int x = 0; x < d; x++) {
                Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];
            }
            // 每一行都有一个值，与q对应（gemm结果矩阵与A矩阵(即q)的行数一致）
            float row_m_prev = m[lm_offset + (tile_q * i) + tx];
            float row_l_prev = l[lm_offset + (tile_q * i) + tx];

            // 三层循环计算gemm，其中一层循环用tx并行代替。
            // Q[i,x] * K[y,x] 
            // for (i:tile_q)     // i (tile_q==tx数量，省去该循环)
            //    for (j:tile_kv) // j
            //       for (x:d)    // k

            // S = QK^T, row_m = rowmax(S)
            float row_m = -INFINITY;
            for (int y = 0; y < tile_kv; y++) {
                float sum = 0;
                // k层循环计算完，C(i,j)即是结果，需要除以缩放因子，如sqrt(d)
                for (int x = 0; x < d; x++) {
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                }
                sum *= softmax_scale;
                // gemm中tx对应i，所以S也用tx充当i，j对应y。x是k维度，不存在S里
                // row_m拿到的是S中每一行的最大值，tx对应行号，每个tx都有一个最大值row_m
                S[(tx * tile_kv) + y] = sum;
                if (sum > row_m)
                    row_m = sum;
            }

            // S里每个数值都进行相同的操作，row_m是S每一行的最大值，row_l是S减最大值并做exp后的累计值
            // P = exp(S - row_m), row_l = rowsum(P)
            float row_l = 0;
            for (int y = 0; y < tile_kv; y++) {
                S[(tx * tile_kv) + y] = __expf(S[(tx * tile_kv) + y] - row_m);
                row_l += S[(tx * tile_kv) + y];
            }

            // 更新row_m，直接取更大值。
            // exp(pre-new)*pre_l + exp(cur-new)*cur_l
            // Compute new m and l
            float row_m_new = max(row_m_prev, row_m);
            float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);

            // p[tile_q, tile_kv] * v[tile_kv, d] = pv[tile_q, d]
            // 下面同样tx代替i循环，而x对应j循环，y对应k循环tile_kv。
            // Write O, l, m to HBM
            for (int x = 0; x < d; x++) {
                float pv = 0;  // Pij * Vj
                for (int y = 0; y < tile_kv; y++) {
                    pv += S[(tx * tile_kv) + y] * Vj[(y * d) + x];
                }
                // (1/row_l_new) * (历史KV块贡献 + 当前kv块贡献)
                float old_value = O[qkv_offset + (tile_size * i) + (tx * d) + x];
                O[qkv_offset + (tile_size * i) + (tx * d) + x] = (1 / row_l_new) * 
                    ((row_l_prev * __expf(row_m_prev - row_m_new) * old_value) + (__expf(row_m - row_m_new) * pv));
            }
            m[lm_offset + (tile_q * i) + tx] = row_m_new;
            l[lm_offset + (tile_q * i) + tx] = row_l_new;
        }
        __syncthreads();  // otherwise, thread can use the wrong Kj, Vj in inner loop
    }
}
