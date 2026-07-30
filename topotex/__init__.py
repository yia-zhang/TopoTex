# -*- coding: utf-8 -*-
"""TOPOTEX — topology-indexed texture generation.

Public API:
    Image + Mesh -> Face Set Latent Z_F -> UV Query -> Flow Matching
    -> Texture

    pipeline = TopoTexPipeline.from_checkpoint("checkpoints/baseline")
    result = pipeline(mesh=item.mesh, uv_query=item.uv_queries[0],
                      images=item.mv_images, num_steps=50, seed=20260727)
"""

from topotex.config import (
    DataConfig,
    FlowMatchingConfig,
    SurfaceConditionerConfig,
    TopoTexConfig,
    TrainingConfig,
)
from topotex.data.dataset import TopoTexDataset
from topotex.data.schema import (
    ConditionerOutput,
    FaceGraph,
    MeshBatch,
    TopoTexBatch,
    TopoTexOutput,
    UVQueryBatch,
)
from topotex.models.topotex import TopoTexModel, build_models
from topotex.pipelines.inference import TopoTexPipeline

__all__ = [
    "TopoTexModel",
    "TopoTexPipeline",
    "TopoTexConfig",
    "TopoTexBatch",
    "UVQueryBatch",
    "MeshBatch",
    "FaceGraph",
    "ConditionerOutput",
    "TopoTexOutput",
    "TopoTexDataset",
    "build_models",
    "DataConfig",
    "SurfaceConditionerConfig",
    "FlowMatchingConfig",
    "TrainingConfig",
]
