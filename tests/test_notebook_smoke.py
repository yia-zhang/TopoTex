# -*- coding: utf-8 -*-
"""Notebook smoke: model import, dataset loading, one-sample forward, and
each maintained notebook executes end to end (inference only — no training
happens in any notebook)."""

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROJECT = Path(__file__).resolve().parents[1]
DATASET = PROJECT / "output" / "topotex_dataset"
HAVE_DATA = (DATASET / "manifest.jsonl").exists()
HAVE_CKPT = (PROJECT / "checkpoints" / "baseline" / "ckpt.pt").exists()
HAVE_CUDA = torch.cuda.is_available()
NOTEBOOKS = [
    "Dataset_Inspector.ipynb",
    "Model_Inspector.ipynb",
    "Technical_Report.ipynb",
    "Pipeline_Playground.ipynb",
]


def test_model_import_and_forward():
    """Model packages import and a tiny conditioner forward runs on CPU."""
    from topotex.models import (  # noqa
        MaskedFlowMatching,
        MiniDiT,
        SurfaceConditioner,
    )

    torch.manual_seed(0)
    model = SurfaceConditioner(image_size=64, resolution=48).eval()
    V = torch.tensor(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [2.1, 1.7, 0.6]], dtype=torch.float32
    )
    F = torch.tensor([[0, 1, 2], [1, 3, 2]])
    fid = torch.full((48, 48), -1, dtype=torch.long)
    fid[8:40, 8:40] = 0
    bary = torch.full((48, 48, 3), 1 / 3)
    with torch.no_grad():
        out = model(
            {"vertices": V, "faces": F},
            torch.rand(1, 6, 3, 64, 64),
            {"face_id": fid, "barycentric": bary},
        )
    assert out["uv_condition"].shape == (1, 64, 48, 48)
    assert torch.isfinite(out["uv_condition"]).all()


@pytest.mark.skipif(not HAVE_DATA, reason="dataset not built")
def test_dataset_loading():
    from topotex.data.dataset import TopoTexDataset

    ids = [
        json.loads(l)["sample_id"] for l in open(DATASET / "manifest.jsonl")
    ][:1]
    it = TopoTexDataset(DATASET, ids)[0]
    assert len(it["uv_queries"]) == 3 and len(it["test_uv_queries"]) == 1


@pytest.mark.skipif(
    not (HAVE_DATA and HAVE_CKPT and HAVE_CUDA),
    reason="dataset/checkpoint/GPU unavailable",
)
def test_one_sample_conditioner_forward():
    """Real checkpoint, real sample: encode Z_F and decode one UV query."""
    from topotex.data.dataset import TopoTexDataset
    from topotex.models.topotex import build_models

    ck = torch.load(
        PROJECT / "checkpoints" / "baseline" / "ckpt.pt",
        map_location="cuda:0",
        weights_only=False,
    )
    conditioner, _ = build_models(ck["config"], "cuda:0")
    conditioner.load_state_dict(ck["conditioner"])
    conditioner.eval()
    it = TopoTexDataset(DATASET, ck["samples"][:1], device="cuda:0")[0]
    q = it["uv_queries"][0]
    with torch.no_grad():
        Z_F, _ = conditioner.encode_faces(
            it["mesh"], it["mv_images"].float()[None] / 255, it["graph"]
        )
        out = conditioner(
            it["mesh"],
            None,
            {
                "face_id": q["face_id"],
                "barycentric": q["barycentric"].permute(1, 2, 0),
            },
            face_tokens=Z_F,
        )
    assert Z_F.shape == (len(it["mesh"]["faces"]), 256)
    assert out["uv_condition"].shape[1:] == (64, 256, 256)


@pytest.mark.skipif(
    not (HAVE_DATA and HAVE_CKPT and HAVE_CUDA),
    reason="dataset/checkpoint/GPU unavailable",
)
@pytest.mark.parametrize("nb_name", NOTEBOOKS)
def test_notebook_executes(nb_name, tmp_path):
    """Execute each maintained notebook against the real data/checkpoint.
    Runs inference (DDIM sampling) but never training."""
    import nbformat
    from nbclient import NotebookClient

    nb_path = PROJECT / "notebooks" / nb_name
    nb = nbformat.read(nb_path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=1200,
        kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT / "notebooks")}},
    )
    client.execute()  # raises CellExecutionError on any failure
