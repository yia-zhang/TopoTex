# -*- coding: utf-8 -*-
"""Minimal frozen UniTEX stage-1 adapter for the dataset pipeline.

reference image + textureless mesh -> MV generation -> delight -> six views
(uint8 [6,3,res,res], fixed order front/back/left/right/top/bottom).

The LTM/texture stages are never constructed (build_ltm is stubbed before
pipeline init); SR is disabled. View mapping [0,3,1,4,2,5] was verified
empirically via CCM-centroid matching (UniTEX's left/right labels are
mirrored w.r.t. ours).
"""
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UNITEX_ROOT = Path(os.environ.get("UNITEX_ROOT",
                                  "/root/youjiaZhang/UniTEX/upstream"))
VIEW_ORDER = ["front", "back", "left", "right", "top", "bottom"]
UNITEX_GRID_TO_OURS = [0, 3, 1, 4, 2, 5]   # frtbld grid slot per our view
GENERATOR_NAME = "unitex_flux_mv"
BACKGROUND = 235
MV_LORA = ("lyxun/UniTEX", "mv_lora_weights.safetensors")
DELIGHT_LORA = ("lyxun/UniTEX", "delight_lora_weights.safetensors")


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


class UniTexMV:
    GEN_STEPS = 28   # hard-coded in upstream build_pipeline

    def __init__(self, seed=20260727, resolution=256):
        self.seed = seed
        self.resolution = resolution
        self._pipe = None
        self._rembg = None
        self._ckpt_sha = None

    def checkpoint_sha256(self):
        """Combined digest of both LoRAs shaping the output."""
        if self._ckpt_sha is None:
            from huggingface_hub import hf_hub_download
            h1 = sha256_file(hf_hub_download(*MV_LORA))
            h2 = sha256_file(hf_hub_download(*DELIGHT_LORA))
            self._ckpt_sha = hashlib.sha256(f"{h1}:{h2}".encode()).hexdigest()
        return self._ckpt_sha

    def _load(self):
        if self._pipe is not None:
            return
        sys.path.insert(0, str(UNITEX_ROOT))
        os.chdir(UNITEX_ROOT)   # upstream uses relative resource paths
        import pipeline as unitex_pipeline
        # hard guarantee: the LTM/texture-field is never constructed
        unitex_pipeline.build_ltm = lambda *a, **k: (None, True, False)
        self._pipe = unitex_pipeline.CustomRGBTextureFullPipeline(
            super_resolutions=False, filt_gradient_points=False,
            filt_large_angle_points=True, seed=self.seed)

    def _load_prep(self):
        """Background removal + geometry-control renderer only (no FLUX)."""
        if self._pipe is not None:
            self._rembg = self._pipe.rembg_session
            self._vexp = self._pipe.video_exporter
            return
        if self._rembg is not None:
            return
        sys.path.insert(0, str(UNITEX_ROOT))
        os.chdir(UNITEX_ROOT)
        import pipeline as unitex_pipeline
        self._rembg = unitex_pipeline.RMBG2(pretrain_models=None)
        from TextureTools.texturetools.video.export_nvdiffrast_video import \
            VideoExporter
        self._vexp = VideoExporter()

    def _sample_seed(self, sample_id):
        return int.from_bytes(hashlib.sha256(
            f"{self.seed}:{sample_id}".encode()).digest()[:4], "little")

    def prepare_inputs(self, reference_image, textureless_mesh, sample_id,
                       out_dir):
        """Processed reference + geometry controls (FLUX/seed independent)."""
        from PIL import Image
        import torch
        reference_image = Path(reference_image).resolve()
        textureless_mesh = Path(textureless_mesh).resolve()
        out_dir = Path(out_dir).resolve()
        self._load_prep()
        from TextureTools.texturetools.image.process_image import preprocess
        out_dir.mkdir(parents=True, exist_ok=True)
        img = Image.open(reference_image).convert("RGB").resize((1024, 1024))
        out = preprocess(img, alpha=None, H=1024, W=1024, scale=0.95,
                         color="grey", return_alpha=False,
                         rembg_session=self._rembg)
        out.convert("RGB").resize((512, 512)).save(
            out_dir / "processed_image.png")
        cond = self._vexp.export_condition(
            str(textureless_mesh), geometry_scale=0.95, n_views=6, n_rows=2,
            n_cols=3, H=512, W=512, fov_deg=49.1, scale=1.0,
            perspective=False, orbit=False, background="grey",
            return_info=False, return_image=True, return_mesh=False,
            return_camera=True)
        cond["alpha"].save(out_dir / "mv_alpha.png")
        cond["ccm"].save(out_dir / "mv_ccm.png")
        cond["normal"].save(out_dir / "mv_normal.png")

    def generate(self, inputs_dir, sample_id, work_dir):
        """FLUX MV + delight from prepared controls -> six-view tensor."""
        from PIL import Image
        import torch
        inputs_dir = Path(inputs_dir).resolve()
        work = Path(work_dir).resolve()
        self._load()
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
        seed = self._sample_seed(sample_id)
        self._pipe.generator = torch.Generator("cuda").manual_seed(seed)
        t0 = time.time()
        self._pipe.infer_mv(str(work),
                            str(inputs_dir / "processed_image.png"),
                            str(inputs_dir / "mv_normal.png"),
                            str(inputs_dir / "mv_ccm.png"))
        gen_s = time.time() - t0
        rgb = np.asarray(Image.open(work / "mv_rgb.png").convert("RGB"))
        alpha = np.asarray(Image.open(inputs_dir / "mv_alpha.png")
                           .convert("L"))
        H, W = alpha.shape
        hp, wp = H // 2, W // 3
        grid_rgb = rgb.reshape(2, hp, 3, wp, 3).transpose(0, 2, 1, 3, 4) \
                      .reshape(6, hp, wp, 3)
        grid_a = alpha.reshape(2, hp, 3, wp).transpose(0, 2, 1, 3) \
                      .reshape(6, hp, wp).astype(np.float32) / 255.0
        res = self.resolution
        out = np.zeros((6, 3, res, res), np.uint8)
        for k, slot in enumerate(UNITEX_GRID_TO_OURS):
            im = grid_rgb[slot].astype(np.float32)
            a = grid_a[slot][..., None]
            comp = im * a + BACKGROUND * (1 - a)
            img = Image.fromarray(comp.astype(np.uint8)).resize(
                (res, res), Image.LANCZOS)
            out[k] = np.asarray(img).transpose(2, 0, 1)
        meta = {"generator_name": GENERATOR_NAME,
                "generator_checkpoint_sha256": self.checkpoint_sha256(),
                "generation_steps": self.GEN_STEPS,
                "generation_seed": seed,
                "generation_time_s": round(gen_s, 1),
                "grid_to_ours": UNITEX_GRID_TO_OURS,
                "background": BACKGROUND}
        return out, meta
