# -*- coding: utf-8 -*-
"""TopoTexModel: the composed method — Surface Conditioner (Z_F encoder +
Global UV Query Attention decoder) + rectified-flow generator.

Checkpoint layout is FROZEN: ``{"conditioner": state_dict, "dit":
state_dict, "config": flat dict, ...}`` — :meth:`TopoTexModel.
from_checkpoint` loads it without any migration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn

from topotex.config import TopoTexConfig
from topotex.data.schema import (
    ConditionerOutput,
    FaceGraph,
    MeshBatch,
    UVQueryBatch,
)
from topotex.layers.flow import MaskedFlowMatching
from topotex.models.flow_matching import MiniDiT
from topotex.models.surface_conditioner import SurfaceConditioner


def build_models(cfg: dict, device: str):
    """Frozen factory: flat config dict -> (conditioner, velocity net).

    Seeds torch with ``cfg["seed"]`` before construction (part of the
    frozen recipe — initial weights are reproducible).
    """
    torch.manual_seed(int(cfg["seed"]))
    conditioner = SurfaceConditioner(
        dim=int(cfg["cond_dim"]),
        out_channels=int(cfg["cond_channels"]),
        pe_kind="random_walk",
        pe_k=int(cfg["pe_k"]),
        heads=int(cfg["cond_heads"]),
        cross_depth=int(cfg["cross_depth"]),
        topo_depth=int(cfg["topo_depth"]),
        query_depth=int(cfg["query_depth"]),
        image_size=int(cfg["image_size"]),
        resolution=int(cfg["resolution"]),
        patch=int(cfg["patch"]),
    ).to(device)
    dit = MiniDiT(
        resolution=int(cfg["resolution"]),
        patch=int(cfg["patch"]),
        hidden=int(cfg["dit_hidden"]),
        depth=int(cfg["dit_depth"]),
        heads=int(cfg["dit_heads"]),
        mlp_ratio=4.0,
        cond_channels=int(cfg["cond_channels"]),
    ).to(device)
    return conditioner, dit


class TopoTexModel(nn.Module):
    """conditioner + velocity net + flow schedule, as one module.

    Submodule names ("conditioner", "dit") match the frozen checkpoint's
    top-level state-dict keys.
    """

    def __init__(
        self, conditioner: SurfaceConditioner, dit: MiniDiT, config: dict
    ):
        super().__init__()
        self.conditioner = conditioner
        self.dit = dit
        self.config = dict(config)
        self.schedule = MaskedFlowMatching(
            T=int(config["T"]), device=next(dit.parameters()).device.type
        )

    # ------------------------------------------------------------ factory
    @classmethod
    def from_config(
        cls, config: Union[dict, TopoTexConfig], device: str = "cuda:0"
    ) -> "TopoTexModel":
        flat = (
            config.to_dict()
            if isinstance(config, TopoTexConfig)
            else dict(config)
        )
        TopoTexConfig.from_dict(flat)  # validate (raises loudly)
        conditioner, dit = build_models(flat, device)
        return cls(conditioner, dit, flat)

    @classmethod
    def from_checkpoint(cls, path: Union[str, Path], device: str = "cuda:0"):
        """Load a frozen checkpoint. Returns (model, checkpoint_dict)."""
        path = Path(path)
        if path.is_dir():
            path = path / "ckpt.pt"
        if not path.exists():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        ck = torch.load(path, map_location=device, weights_only=False)
        model = cls.from_config(ck["config"], device)
        model.conditioner.load_state_dict(ck["conditioner"])
        model.dit.load_state_dict(ck["dit"])
        model.eval()
        return model, ck

    # ------------------------------------------------------------- stages
    @torch.no_grad()
    def encode(
        self,
        mesh: MeshBatch,
        images: torch.Tensor,
        graph: Optional[FaceGraph] = None,
    ) -> torch.Tensor:
        """mesh + six views -> Z_F.

        images: uint8/float [Nv, 3, S, S] (uint8 is scaled to [0,1]).
        Returns float32 [F, D].
        """
        if images.dtype == torch.uint8:
            images = images.float() / 255
        z, _ = self.conditioner.encode_faces(mesh, images[None], graph)
        return z

    @torch.no_grad()
    def condition(
        self, face_tokens: torch.Tensor, uv_query: UVQueryBatch
    ) -> ConditionerOutput:
        """Read Z_F through one UV query -> dense uv_condition."""
        return self.conditioner(
            None,
            None,
            {
                "face_id": uv_query["face_id"],
                "barycentric": uv_query["barycentric"].permute(1, 2, 0),
            },
            face_tokens=face_tokens,
        )

    @torch.no_grad()
    def generate(
        self,
        uv_condition: torch.Tensor,
        valid_mask: torch.Tensor,
        num_steps: int = 50,
        seed: int = 20260727,
    ) -> torch.Tensor:
        """Euler-integrate the velocity field -> texture [3, H, W] in
        [-1, 1] (0 outside the valid mask)."""
        device = uv_condition.device
        g = torch.Generator(device=device).manual_seed(seed)
        return self.schedule.ddim_sample(
            self.dit,
            uv_condition,
            valid_mask.float()[None, None],
            steps=num_steps,
            generator=g,
        )[0]
