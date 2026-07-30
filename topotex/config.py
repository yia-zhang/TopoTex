# -*- coding: utf-8 -*-
"""Typed configuration.

YAML files under ``configs/`` remain the experiment source of truth; they
are validated into these dataclasses at load time. Model constructors keep
plain keyword arguments — configs are unpacked at the factory boundary
(:func:`topotex.models.topotex.build_models`), never passed through.

The flat key set is FROZEN: it is what training writes into every
checkpoint (``ckpt["config"]``), so existing checkpoints round-trip through
:meth:`TopoTexConfig.from_dict` unchanged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DataConfig:
    """Dataset location and query sampling."""

    dataset_root: str = "output/topotex_dataset"
    #: draw probabilities for (canonical, alternative, partial) queries
    query_probs: tuple[float, float, float] = (0.5, 0.3, 0.2)

    def validate(self) -> None:
        if abs(sum(self.query_probs) - 1.0) > 1e-6:
            raise ValueError(
                f"query_probs must sum to 1, got {self.query_probs}"
            )


@dataclass
class SurfaceConditionerConfig:
    """Face-set encoder + Global UV Query Attention decoder."""

    cond_dim: int = 256  #: face-token width D
    cond_channels: int = 64  #: dense uv_condition channels C
    pe_k: int = 16  #: random-walk PE steps
    cond_heads: int = 8
    cross_depth: int = 2  #: face-image cross-attention blocks
    topo_depth: int = 4  #: sparse topology-transformer blocks
    query_depth: int = 4  #: UV query cross-attention blocks
    image_size: int = 256  #: conditioning view resolution

    def validate(self) -> None:
        if self.cond_dim % self.cond_heads:
            raise ValueError(
                f"cond_dim {self.cond_dim} % cond_heads {self.cond_heads}"
            )


@dataclass
class FlowMatchingConfig:
    """Rectified-flow generator (MiniDiT velocity net + schedule)."""

    resolution: int = 256
    patch: int = 8
    dit_hidden: int = 384
    dit_depth: int = 8
    dit_heads: int = 6
    T: int = 1000  #: integer flow-time scale (tau = t/T)
    generator: str = "fm"

    def validate(self) -> None:
        if self.resolution % self.patch:
            raise ValueError("resolution must be divisible by patch")
        if self.dit_hidden % self.dit_heads:
            raise ValueError(
                f"dit_hidden {self.dit_hidden} % dit_heads {self.dit_heads}"
            )
        if self.generator != "fm":
            raise ValueError("flow matching is the only generator")


@dataclass
class TrainingConfig:
    """Frozen training recipe (budget in mesh exposures)."""

    seed: int = 20260729
    target_mesh_exposures: int = 2000
    noise_batch: int = 4
    lr: float = 3.0e-4
    warmup: int = 100
    lr_final_frac: float = 0.1
    t_high_frac: float = 0.5  #: fraction of noise batch drawn from high t
    t_high_min: int = 700
    aux_rgb_weight: float = 0.1
    log_every: int = 200
    ckpt_every: int = 2000
    group_size: int = 4  #: meshes packed per face graph
    precision: str = "fp32"  #: "bf16" enables autocast on the loss

    def validate(self) -> None:
        if self.precision not in ("fp32", "bf16"):
            raise ValueError(f"precision {self.precision!r}")
        if not 0.0 <= self.t_high_frac <= 1.0:
            raise ValueError("t_high_frac in [0,1]")


@dataclass
class TopoTexConfig:
    """Aggregate config = the checkpoint's flat ``config`` dict, typed."""

    data: DataConfig = field(default_factory=DataConfig)
    conditioner: SurfaceConditionerConfig = field(
        default_factory=SurfaceConditionerConfig
    )
    flow: FlowMatchingConfig = field(default_factory=FlowMatchingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    #: keys outside the typed schema (e.g. world_size/steps stamped by the
    #: trainer) — preserved verbatim for checkpoint round-tripping
    extra: dict[str, Any] = field(default_factory=dict)

    _FIELD_MAP = None  # class-level cache

    @classmethod
    def _field_map(cls) -> dict[str, str]:
        if cls._FIELD_MAP is None:
            m = {}
            for group in ("data", "conditioner", "flow", "training"):
                proto = cls.__dataclass_fields__[group].default_factory()
                for k in asdict(proto):
                    m[k] = group
            cls._FIELD_MAP = m
        return cls._FIELD_MAP

    @classmethod
    def from_dict(cls, flat: dict[str, Any]) -> "TopoTexConfig":
        """Validate a flat YAML/checkpoint dict into typed groups."""
        groups: dict[str, dict[str, Any]] = {
            "data": {},
            "conditioner": {},
            "flow": {},
            "training": {},
        }
        extra: dict[str, Any] = {}
        fmap = cls._field_map()
        for k, v in flat.items():
            if k in fmap:
                if k == "query_probs":
                    v = tuple(float(x) for x in v)
                groups[fmap[k]][k] = v
            else:
                extra[k] = v
        cfg = cls(
            data=DataConfig(**groups["data"]),
            conditioner=SurfaceConditionerConfig(**groups["conditioner"]),
            flow=FlowMatchingConfig(**groups["flow"]),
            training=TrainingConfig(**groups["training"]),
            extra=extra,
        )
        cfg.validate()
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TopoTexConfig":
        import yaml

        flat = yaml.safe_load(open(path))
        if not isinstance(flat, dict):
            raise ValueError(f"{path}: not a mapping")
        return cls.from_dict(flat)

    def to_dict(self) -> dict[str, Any]:
        """Back to the frozen flat form (checkpoint `config` layout)."""
        flat: dict[str, Any] = {}
        for group in (self.data, self.conditioner, self.flow, self.training):
            d = asdict(group)
            if "query_probs" in d:
                d["query_probs"] = list(d["query_probs"])
            flat.update(d)
        flat.update(self.extra)
        return flat

    def validate(self) -> None:
        self.data.validate()
        self.conditioner.validate()
        self.flow.validate()
        self.training.validate()
