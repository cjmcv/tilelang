import os
import ctypes

_this_dir = os.path.dirname(__file__)

from .core import *
from .kernel import *
from .threadblock import *
from .persistent_kernel import PersistentKernel

__version__ = "0.2.4"

def new_kernel_graph():
    kgraph = core.CyKNGraph()
    return KNGraph(kgraph)

