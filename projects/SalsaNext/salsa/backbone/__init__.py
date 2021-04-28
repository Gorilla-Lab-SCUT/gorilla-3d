# Copyright (c) Gorilla-Lab. All rights reserved.
from .darknet import DarkNet
from .squeezenet import SqueezeNet
from .squeezenetV2 import SqueezeNetV2

# auto registry all defined module
import torch
import gorilla
gorilla.core.auto_registry(gorilla.MODULES, globals(), type=torch.nn.Module)
