# Copyright (c) Gorilla-Lab. All rights reserved.
from .shapenet_part import ShapeNetPartNormal

__all__ = [k for k in globals().keys() if not k.startswith("_")]
