import os
from tvm.tir import stmt_functor, Block, For


def save_to_file(text: str, file_path: str) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)
        
def get_smem_bytes(func):
    smem_bytes = 0
    num_stages = 1
    
    def collect(node):
        nonlocal smem_bytes
        nonlocal num_stages
        if isinstance(node, Block):
            for buf in node.alloc_buffers:
                scope = buf.scope()
                if str(scope).startswith("shared"):
                    numel = 1
                    for s in buf.shape:
                        numel *= int(s)
                    smem_bytes += numel * (buf.dtype.bits // 8)
        if isinstance(node, For):
            num_stages = node.annotations.get("num_stages", 1)
                
    stmt_functor.post_order_visit(func.body, collect)
    return smem_bytes*num_stages