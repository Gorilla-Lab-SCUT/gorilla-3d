# Copyright (c) Gorilla-Lab. All rights reserved.
from .votenet import VoteNet
from .deepsdf import DeepSDF

from .pointnet import (PointNetCls, PointNetSeg)
from .dgcnn import (DGCNNCls, DGCNNPartSeg, DGCNNSemSeg)

from gorilla import MODELS, auto_registry
auto_registry(MODELS, globals())

__all__ = [k for k in globals().keys() if not k.startswith("_")]
