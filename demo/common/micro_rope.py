import itertools

import tilelang
import tilelang.language as T

from common.micro_base import BaseMicroKernel, HparamSelectMode
    
class _RopeStrategy:
    def __init__(self, batch, seq_len, num_heads_q, num_heads_k, head_dim, dtype, accum_dtype):
        self.name_suffix = f"_{batch}_{seq_len}_{num_heads_q}_{num_heads_k}_{head_dim}"
        self.name = "rope_tl"+self.name_suffix
        self.batch = batch  
        self.seq_len = seq_len  
        self.num_heads_q = num_heads_q
        self.num_heads_k = num_heads_k
        self.head_dim = head_dim
        
        self.dtype = dtype
        self.accum_dtype = accum_dtype
        
        self.hparam_space = self._get_hparam_space()
        print(len(self.hparam_space))
        
    def _get_hparam_space(self):
        BLOCK_SEQ=[1]
        BLOCK_HEADS_Q=[1]
        BLOCK_HEADS_K=[1]
        thread_nums=[128]
        
        res = []
        for seq, head_q, head_k, thread_num in itertools.product(
           BLOCK_SEQ, BLOCK_HEADS_Q, BLOCK_HEADS_K, thread_nums):
            res.append([seq, head_q, head_k, thread_num])
        return res 
    
    def get_heuristic_hparams(self):
        # [BLOCK_SEQ, BLOCK_HEADS_Q, BLOCK_HEADS_K, thread_nums]
        return [1,1,1,128]
        
    def get_kernel(self, selected_hparams):
        print("selected_hparams: ", selected_hparams)
        return self.rope_qk_parallel(self.batch, self.seq_len, 
                                     self.num_heads_q, self.num_heads_k, self.head_dim,
                                     *selected_hparams, self.dtype, self.accum_dtype) 

    @tilelang.jit(out_idx=[-2, -1])
    def rope_qk_overlap(batch, seq_len, num_heads_q, num_heads_k, head_dim, 
                                BLOCK_SEQ, BLOCK_HEADS_Q, BLOCK_HEADS_K, threads=128,
                                dtype="bfloat16", accum_dtype="float32"):
        """
        优化版本：利用cos/sin的前后一半相同的特点
        1. 只加载一半的cos/sin到shared memory（节省50% shared memory）
        2. 减少一半的cos/sin加载操作
        3. 使用相同的cos/sin值计算前后half_dim
        """
        @T.prim_func
        def rope(
            Q: T.Tensor((batch, seq_len, num_heads_q, head_dim), dtype),
            K: T.Tensor((batch, seq_len, num_heads_k, head_dim), dtype),
            cos: T.Tensor((batch, seq_len, head_dim), dtype),
            sin: T.Tensor((batch, seq_len, head_dim), dtype),
            Q_embed: T.Tensor((batch, seq_len, num_heads_q, head_dim), dtype),
            K_embed: T.Tensor((batch, seq_len, num_heads_k, head_dim), dtype),
        ):
            half_dim = head_dim // 2
            
            # 验证cos/sin的前后一半相同（可选，调试用）
            # 注意：在实际产品代码中可能不需要这个断言
            # T.Assert(T.all(cos[..., :half_dim] == cos[..., half_dim:]), "cos前后一半应该相同")
            # T.Assert(T.all(sin[..., :half_dim] == sin[..., half_dim:]), "sin前后一半应该相同")
            
            max_head_blocks = T.max(
                T.ceildiv(num_heads_q, BLOCK_HEADS_Q),
                T.ceildiv(num_heads_k, BLOCK_HEADS_K)
            )
            
            with T.Kernel(batch, 
                        max_head_blocks,
                        T.ceildiv(seq_len, BLOCK_SEQ),
                        threads=threads) as (bx, by, bz):
                
                batch_id = bx
                seq_start = bz * BLOCK_SEQ

                # 优化1：只存储一半的cos/sin到shared memory
                cos_sh = T.alloc_shared((BLOCK_SEQ, half_dim), dtype)  # 节省50%
                sin_sh = T.alloc_shared((BLOCK_SEQ, half_dim), dtype)  # 节省50%
                
                # 1. 加载cos/sin（只加载前一半）
                for s, d in T.Parallel(BLOCK_SEQ, half_dim):
                    global_seq = seq_start + s
                    if global_seq < seq_len:
                        cos_sh[s, d] = cos[batch_id, global_seq, d]  # 只取前half_dim
                        sin_sh[s, d] = sin[batch_id, global_seq, d]  # 只取前half_dim
                
                # 2. 处理Q（如果当前head块对应Q）
                head_q_start = by * BLOCK_HEADS_Q
                if head_q_start < num_heads_q:
                    # 优化2：可以只存储Q的一半，计算时重组
                    # 但为了简单，我们先保持完整存储
                    Q_sh = T.alloc_shared((BLOCK_SEQ, BLOCK_HEADS_Q, head_dim), dtype)
                    
                    # 加载Q数据
                    for s, h, d in T.Parallel(BLOCK_SEQ, BLOCK_HEADS_Q, head_dim):
                        global_seq = seq_start + s
                        global_head = head_q_start + h
                        if global_seq < seq_len and global_head < num_heads_q:
                            Q_sh[s, h, d] = Q[batch_id, global_seq, global_head, d]
                    
                    # 计算Q的RoPE（使用优化后的cos/sin）
                    for s, h, d in T.Parallel(BLOCK_SEQ, BLOCK_HEADS_Q, half_dim):
                        global_seq = seq_start + s
                        global_head = head_q_start + h
                        
                        if global_seq < seq_len and global_head < num_heads_q:
                            # 读取Q数据
                            a_q = Q_sh[s, h, d].astype(accum_dtype)          # 前一半
                            b_q = Q_sh[s, h, d + half_dim].astype(accum_dtype)  # 后一半
                            
                            # 关键优化：使用相同的cos/sin值（只从cos_sh取前half_dim）
                            cos_val = cos_sh[s, d].astype(accum_dtype)
                            sin_val = sin_sh[s, d].astype(accum_dtype)
                            
                            # 前后half_dim使用相同的cos/sin值
                            # 前half_dim: a*cos - b*sin
                            # 后half_dim: b*cos + a*sin
                            out_first_q = a_q * cos_val - b_q * sin_val
                            out_second_q = b_q * cos_val + a_q * sin_val
                            
                            Q_embed[batch_id, global_seq, global_head, d] = out_first_q.astype(dtype)
                            Q_embed[batch_id, global_seq, global_head, d + half_dim] = out_second_q.astype(dtype)
                
                # 3. 处理K（如果当前head块对应K）
                head_k_start = by * BLOCK_HEADS_K
                if head_k_start < num_heads_k:
                    K_sh = T.alloc_shared((BLOCK_SEQ, BLOCK_HEADS_K, head_dim), dtype)
                    
                    # 加载K数据
                    for s, h, d in T.Parallel(BLOCK_SEQ, BLOCK_HEADS_K, head_dim):
                        global_seq = seq_start + s
                        global_head = head_k_start + h
                        if global_seq < seq_len and global_head < num_heads_k:
                            K_sh[s, h, d] = K[batch_id, global_seq, global_head, d]
                    
                    # 计算K的RoPE（使用相同的cos/sin）
                    for s, h, d in T.Parallel(BLOCK_SEQ, BLOCK_HEADS_K, half_dim):
                        global_seq = seq_start + s
                        global_head = head_k_start + h
                        
                        if global_seq < seq_len and global_head < num_heads_k:
                            # 读取K数据
                            a_k = K_sh[s, h, d].astype(accum_dtype)
                            b_k = K_sh[s, h, d + half_dim].astype(accum_dtype)
                            
                            # 使用相同的cos/sin值
                            cos_val = cos_sh[s, d].astype(accum_dtype)
                            sin_val = sin_sh[s, d].astype(accum_dtype)
                            
                            # K的RoPE公式
                            out_first_k = a_k * cos_val - b_k * sin_val
                            out_second_k = b_k * cos_val + a_k * sin_val
                            
                            K_embed[batch_id, global_seq, global_head, d] = out_first_k.astype(dtype)
                            K_embed[batch_id, global_seq, global_head, d + half_dim] = out_second_k.astype(dtype)
        
        return rope

    @tilelang.jit(out_idx=[-2, -1])
    def rope_qk_parallel(batch, seq_len, num_heads_q, num_heads_k, head_dim, 
                        BLOCK_SEQ, BLOCK_HEADS_Q, BLOCK_HEADS_K, threads=128,
                        dtype="bfloat16", accum_dtype="float32"):
        """
        使用T.macro提取公共计算逻辑的版本
        """
        half_dim = head_dim // 2
        q_shape = [batch, seq_len, num_heads_q, head_dim]
        k_shape = [batch, seq_len, num_heads_k, head_dim]
        
        @T.macro
        def ComputeRoPE(
            data_ptr: T.Tensor,          # Q或K的输入张量
            cos_ptr: T.Tensor,           # cos张量
            sin_ptr: T.Tensor,           # sin张量  
            out_ptr: T.Tensor,           # 输出张量
            batch_id: T.int32,           # batch索引
            head_start: T.int32,         # head起始索引
            seq_start: T.int32,          # seq起始索引
            num_heads: T.int32,          # head总数
            block_heads: T.int32,        # 每个block的head数
            data_sh: T.SharedBuffer,     # 数据shared memory
            cos_sh: T.SharedBuffer,      # cos shared memory
            sin_sh: T.SharedBuffer,      # sin shared memory
        ):
            """RoPE计算的宏"""
            # 加载cos/sin到shared memory
            for s, d in T.Parallel(BLOCK_SEQ, half_dim):
                global_seq = seq_start + s
                if global_seq < seq_len:
                    cos_sh[s, d] = cos_ptr[batch_id, global_seq, d]
                    sin_sh[s, d] = sin_ptr[batch_id, global_seq, d]
            
            # 加载数据到shared memory
            for s, h, d in T.Parallel(BLOCK_SEQ, block_heads, head_dim):
                global_seq = seq_start + s
                global_head = head_start + h
                if global_seq < seq_len and global_head < num_heads:
                    data_sh[s, h, d] = data_ptr[batch_id, global_seq, global_head, d]
            
            # 计算RoPE并写回
            for s, h, d in T.Parallel(BLOCK_SEQ, block_heads, half_dim):
                global_seq = seq_start + s
                global_head = head_start + h
                
                if global_seq < seq_len and global_head < num_heads:
                    a = data_sh[s, h, d].astype(accum_dtype)
                    b = data_sh[s, h, d + half_dim].astype(accum_dtype)
                    
                    cos_val = cos_sh[s, d].astype(accum_dtype)
                    sin_val = sin_sh[s, d].astype(accum_dtype)
                    
                    out_first = a * cos_val - b * sin_val
                    out_second = b * cos_val + a * sin_val
                    
                    out_ptr[batch_id, global_seq, global_head, d] = out_first.astype(dtype)
                    out_ptr[batch_id, global_seq, global_head, d + half_dim] = out_second.astype(dtype)
        
        @T.prim_func
        def rope(
            Q: T.Tensor(q_shape, dtype),
            K: T.Tensor(k_shape, dtype),
            cos: T.Tensor((batch, seq_len, head_dim), dtype),
            sin: T.Tensor((batch, seq_len, head_dim), dtype),
            Q_embed: T.Tensor(q_shape, dtype),
            K_embed: T.Tensor(k_shape, dtype),
        ):
            # 计算总block数
            total_q_blocks = batch * T.ceildiv(num_heads_q, BLOCK_HEADS_Q) * T.ceildiv(seq_len, BLOCK_SEQ)
            total_k_blocks = batch * T.ceildiv(num_heads_k, BLOCK_HEADS_K) * T.ceildiv(seq_len, BLOCK_SEQ)
            total_blocks = total_q_blocks + total_k_blocks
            
            with T.Kernel(total_blocks, threads=threads) as (bidx,):
                
                cos_sh = T.alloc_shared((BLOCK_SEQ, half_dim), dtype)
                sin_sh = T.alloc_shared((BLOCK_SEQ, half_dim), dtype)
                
                # Q block
                if bidx < total_q_blocks:
                    blocks_per_batch = T.ceildiv(num_heads_q, BLOCK_HEADS_Q) * T.ceildiv(seq_len, BLOCK_SEQ)
                    batch_id = bidx // blocks_per_batch
                    remaining = bidx % blocks_per_batch
                    head_block = remaining // T.ceildiv(seq_len, BLOCK_SEQ)
                    seq_block = remaining % T.ceildiv(seq_len, BLOCK_SEQ)
                    
                    head_start = head_block * BLOCK_HEADS_Q
                    seq_start = seq_block * BLOCK_SEQ
                    
                    if head_start < num_heads_q:
                        Q_sh = T.alloc_shared((BLOCK_SEQ, BLOCK_HEADS_Q, head_dim), dtype)
                        ComputeRoPE(
                            Q, cos, sin, Q_embed,
                            batch_id, head_start, seq_start, num_heads_q, BLOCK_HEADS_Q,
                            Q_sh, cos_sh, sin_sh
                        )
                
                # K block  
                else:
                    k_bidx = bidx - total_q_blocks
                    k_blocks_per_batch = T.ceildiv(num_heads_k, BLOCK_HEADS_K) * T.ceildiv(seq_len, BLOCK_SEQ)
                    k_batch_id = k_bidx // k_blocks_per_batch
                    k_remaining = k_bidx % k_blocks_per_batch
                    k_head_block = k_remaining // T.ceildiv(seq_len, BLOCK_SEQ)
                    seq_block = k_remaining % T.ceildiv(seq_len, BLOCK_SEQ)
                    
                    head_start = k_head_block * BLOCK_HEADS_K
                    seq_start = seq_block * BLOCK_SEQ
                    
                    if head_start < num_heads_k:
                        K_sh = T.alloc_shared((BLOCK_SEQ, BLOCK_HEADS_K, head_dim), dtype)
                        ComputeRoPE(
                            K, cos, sin, K_embed,
                            k_batch_id, head_start, seq_start, num_heads_k, BLOCK_HEADS_K,
                            K_sh, cos_sh, sin_sh
                        )
        return rope

    
class MicroRope(BaseMicroKernel):
    def __init__(self, batch, seq_len, num_heads_q, num_heads_k, head_dim, dtype=T.bfloat16, accum_dtype=T.float32):
        super().__init__()
        self.strategy = _RopeStrategy(batch, seq_len, num_heads_q, num_heads_k, head_dim, dtype, accum_dtype)
        
    def get_source(self, kernel, selected_hparams):
        head_str = \
'''
template <typename T,
          int THREAD_NUM,
          int BLOCK_SEQ,
          int BLOCK_HEADS_Q, 
          int BLOCK_HEADS_K,
          int BATCH,
          int SEQLEN,
          int NUM_HEADS_Q,
          int NUM_HEADS_K,
          int HEAD_DIM>
__device__ __forceinline__ void rope_kernel_<name_suffix>(const int bx, const int by, const int bz,
                                                   const void* __restrict__ q, 
                                                   const void* __restrict__ k, 
                                                   const void* __restrict__ cos,
                                                   const void* __restrict__ sin, 
                                                   void* __restrict__ q_embed_ptr,
                                                   void* __restrict__ k_embed_ptr) {
  static_assert(THREAD_NUM==<threads>);
  static_assert(BLOCK_SEQ==<BLOCK_SEQ>); static_assert(BLOCK_HEADS_Q==<BLOCK_HEADS_Q>); static_assert(BLOCK_HEADS_K==<BLOCK_HEADS_K>);
  static_assert(BATCH==<BATCH>); static_assert(SEQLEN==<SEQLEN>); 
  static_assert(NUM_HEADS_Q==<NUM_HEADS_Q>); static_assert(NUM_HEADS_K==<NUM_HEADS_K>); static_assert(HEAD_DIM==<HEAD_DIM>);
  
  const <dtype>* __restrict__ A = static_cast<const <dtype>*>(input_ptr);
  <dtype>* __restrict__ C = static_cast<<dtype>*>(output_ptr);
  
'''     
        BLOCK_SEQ, BLOCK_HEADS_Q, BLOCK_HEADS_K, threads = selected_hparams
        BLOCK_K = 1
        
        head_str = head_str.replace('<threads>', str(threads))
        head_str = head_str.replace('<BLOCK_SEQ>', str(BLOCK_SEQ))
        head_str = head_str.replace('<BLOCK_HEADS_Q>', str(BLOCK_HEADS_Q)) 
        head_str = head_str.replace('<BLOCK_HEADS_K>', str(BLOCK_HEADS_K)) 
        head_str = head_str.replace('<BATCH>', str(self.strategy.batch))
        head_str = head_str.replace('<SEQLEN>', str(self.strategy.seq_len))
        head_str = head_str.replace('<NUM_HEADS_Q>', str(self.strategy.num_heads_q)) 
        head_str = head_str.replace('<NUM_HEADS_K>', str(self.strategy.num_heads_k)) 
        head_str = head_str.replace('<HEAD_DIM>', str(self.strategy.head_dim)) 
        head_str = head_str.replace('<name_suffix>', self.strategy.name_suffix)
        if self.strategy.dtype == T.bfloat16:
            dtype = "bfloat16_t"
        else:
            dtype = "float16_t"
        head_str = head_str.replace('<dtype>', str(dtype))
                
        origin_source = kernel.get_kernel_source()
        source = self.replace_header(origin_source, "extern \"C\" __global__", 1, head_str)
        source = source.replace("blockIdx.x", "bx")
        source = source.replace("blockIdx.y", "by")
        source = source.replace("blockIdx.z", "bz")
        
        grid_dim, block_dim, dynamic_smem_buf, use_cooperative_groups = kernel.get_launch_info()[0]
        self.layout = f"({grid_dim['blockIdx.x']}, {grid_dim['blockIdx.y']}, {grid_dim['blockIdx.z']}), ({1}, {1}, {1})"
        extra_attr = f"\n// Strategy: {self.strategy.name}"
        extra_attr += f"\n// selected_hparams: {selected_hparams}."
        extra_attr += f"\n// smem: {dynamic_smem_buf} bytes."
        extra_attr += f"\n// use_cooperative_groups: {use_cooperative_groups}."
        extra_attr += f"\n// layout: {self.layout}"
        extra_attr += f"\n// block_dim=({block_dim['threadIdx.x']}, {block_dim['threadIdx.y']}, {block_dim['threadIdx.z']})."
        source += extra_attr
        
        return source
    
    def get_kernel(self, mode: HparamSelectMode):
        kernel, path = self.auto_get_kernel(self.get_source, self.strategy, mode)
        return kernel, path, self.layout