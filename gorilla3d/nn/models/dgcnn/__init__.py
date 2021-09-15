from .dgcnn_cls import DGCNNCls
from .dgcnn_part_seg import DGCNNPartSeg
from .dgcnn_sem_seg import DGCNNSemSeg

__all__ = [k for k in globals().keys() if not k.startswith("_")]
