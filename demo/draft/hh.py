
# type: ignore

Q[batch,      1,    heads, dim]
K[batch, seqlen, kv_heads, dim]
V[batch, seqlen, kv_heads, dim]
M[batch, seqlen, kv_heads]
O[batch,      1,    heads, dim]

def split(Q, K, V, mask, glse, Output_partial):
    tile: [H, N]
    layout: [batch, heads//H, num_split]
    smem: Q_s[H,d], K_s[N,d], V_s[N,d], O_s[H,d]
    frag: acc_s, acc_o, m, l  # max, logsum
    
    bid, hid, sid = blockIdx.xyz
    chunk_size = ceil_div(seqlen, num_split)
    kv_start = sid * chunk_size
    kv_end = min(kv_start + chunk_size, seqlen)
    group_id = (hid * H) // (heads // kv_heads) # heads_per_group
    
    # <opt: async copy Q to smem + prefetch first K/V block>
    copy(Q[bid, 0, hid*H:(hid+1)*H], Q_s)
    init(acc_o=0, l=0, m=-inf)        # <opt: fastmath for inf>
    
    # <opt: multistage, software pipelining across KV blocks to overlap compute & memory>
    for kv in range(kv_start, kv_end, block_N):
        # <opt: coalesced global memory load with boundary check>
        copy(K[bid, kv:kv+N, group_id, :], K_s)
        copy(V[bid, kv:kv+N, group_id, :], V_s)
        gemm(Q_s, K_s^T, acc_s)       # [H,N] <opt: cutlass unit>
        apply_mask(acc_s, mask, kv)   # <opt: early mask loading. vectorize apply(vcmpx)>
        
        m_new = max(m, rowmax(acc_s))  # [H]
        scale = exp(m - m_new)         # [H]  <opt: fastmath exp/log with fma>
        p = exp(acc_s - m_new)         # [H,N]
        l = l * scale + rowsum(p)      # [H]
        acc_o *= scale[:, None]
          
        gemm(p, V_s, acc_o)            # <opt: cutlass unit>
        m = m_new
    
    # <opt: write glse and Output_partial in same coalesced transaction via struct packing>
    acc_o /= l
    glse[bid, hid*H:(hid+1)*H, sid] = log(l) + m
    Output_partial[bid, hid*H:(hid+1)*H, sid, :] = acc_o[:H]


def combine(glse, Output_partial, Output):
    layout: [heads, batch]
    frag: lse
    
    hid, bid = blockIdx.xy
    
    lse[num_split] = glse[bid, hid, :]
    # <opt: tree-reduce max/sum in shared memory to minimize divergence, __shfl_xor_sync>
    l_max = max(lse)
    l_sum = sum(exp(lse - l_max))
    l_global = log(l_sum) + l_max
    
    # <opt: weight reuse across dim>
    # <opt: vectorized load of Output_partial[bid, hid, k, :] and fma with weight>
    o = 0
    for k in range(num_split):
        weight = exp(lse[k] - l_global)
        o += Output_partial[bid, hid, k, :] * weight
    # <opt: coalesced pattern>
    Output[bid, 0, hid, :] = o