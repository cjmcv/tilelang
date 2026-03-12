
from safetensors.torch import load_model
import torch
import torch.distributed as dist
import argparse
import os, json
from common.pkt_util import TorchRef, PerfReporter, Qwen3Info

DEFAULT_SAVE_DIR = os.path.join("outputs", "qwen3")
MAX_SAVE_TOKENS = 100

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-mirage", action="store_true", help="Use Mirage kernels")
    
    torch.set_default_dtype(torch.bfloat16)

    model, tokenizer = Qwen3Info.load_model(0)
    
    hidden_states = torch.randn((1, 1, 1024), dtype=torch.bfloat16, device="cuda")
    attention_mask = None
    position_embeddings = (torch.randn((1, 32768, 128), dtype=torch.bfloat16, device="cuda"), torch.randn((1, 32768, 128), dtype=torch.bfloat16, device="cuda"))
    step = torch.full((1, ), 0, dtype=torch.int32, device="cuda")
    stream = None
    # torch.Size([1, 1, 1024]) None torch.Size([1, 1, 128]) torch.Size([1, 1, 128]) tensor([62], device='cuda:0', dtype=torch.int32)
    print("before forward", hidden_states)
    
    def torch_ref():
        out = model.model.layers[0].forward(hidden_states, attention_mask, position_embeddings, step, stream)
        return out[0]
    
    reporter = PerfReporter() 
    reporter.generate_report(torch_ref, torch_ref, 
                            warnup_iter=100, test_iter=200, 
                            allclose_iter=5, print_mode=1)