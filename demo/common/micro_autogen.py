from .pkt_util import TestUtil, TorchRef
from .micro_base import HparamSelectMode
from .micro_linear import MicroLinearStrategy, MicroLinear
from .micro_rms_norm import MicroRmsNorm
from .micro_silu_mul import MicroSiluMul

class MicroAutoGen:
    def __init__(self, batch_size, hidden_size, intermediate_size):
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        