# Copyright (c) Gorilla-Lab. All rights reserved.
from .darknet import DarkDecoder
from .squeezenet import SqueezeDecoder
from .squeezenetV2 import SqueezeDecoderV2

# auto registry all defined module
import torch
import gorilla
gorilla.core.auto_registry(gorilla.MODULES, globals(), type=torch.nn.Module)
