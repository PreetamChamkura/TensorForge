from tensorforge.passes.shape_propagation import propagate_shapes
from tensorforge.passes.constant_folding import fold_constants
from tensorforge.passes.fusion import fuse_add_relu

__all__ = ["propagate_shapes", "fold_constants", "fuse_add_relu"]
