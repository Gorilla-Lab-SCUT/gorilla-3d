# Copyright (c) Gorilla-Lab. All rights reserved.

from .scannetv2_inst import Dataset

__all__ = [k for k in globals().keys() if not k.startswith("_")]
