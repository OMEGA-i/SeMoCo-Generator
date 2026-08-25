"""Shared backend + skeleton renderer for the SeMoCo-Generator viewers.

`T2MEngine` wraps the trained text2motion model + frozen Flan-T5 encoder + frozen
codec, exposing one-shot and **streaming** text->joints generation plus GT/gen
code decoding. `SkeletonScene` is the reusable viser batched-mesh skeleton
renderer factored out of viser_app.py. Used by:

  * viser_comp_viewer.py  — offline gen-vs-gt comparison over the test split
  * viser_live_viewer.py  — live streaming generation
  * viser_app.py          — rollout player (skeleton rendering only)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Track colors (0-255). Prefix is muted; skeleton tracks use their own color.
PREFIX_COLOR = np.array([154, 160, 166], np.uint8)
GEN_COLOR = np.array([255, 112, 67], np.uint8)     # orange = generated (skeleton)
GT_COLOR = np.array([66, 165, 245], np.uint8)      # blue = ground truth (skeleton)
# Body-mesh default color = kimodo demo's light blue (LIGHT_THEME["mesh"]).
MESH_COLOR = np.array([152, 189, 255], np.uint8)
MESH_GEN_COLOR = np.array([255, 150, 110], np.uint8)   # soft orange (comp: distinguish gen mesh)

# Decoder right receptive field (non-causal codec, stride 4): the last ~88 output
# frames of a prefix decode change as more tokens arrive; earlier frames are final.
STREAM_SETTLE_FRAMES = 88


# ---------------------------------------------------------------------------
# Skeleton rendering (viser batched meshes) — factored from viser_app.py
# ---------------------------------------------------------------------------
def bone_mesh(width: float = 0.09, knuckle: float = 0.12):
    """Unit octahedral 'bone' along +Y (0->1); width scales with bone length."""
    v = np.array([
        [0.0, 0.0, 0.0], [0.0, 1.0, 0.0],
        [width, knuckle, 0.0], [0.0, knuckle, width],
        [-width, knuckle, 0.0], [0.0, knuckle, -width],
    ], np.float32)
    f = np.array([
        [0, 2, 3], [0, 3, 4], [0, 4, 5], [0, 5, 2],
        [1, 3, 2], [1, 4, 3], [1, 5, 4], [1, 2, 5],
    ], np.int32)
    return v, f


def quat_from_y(dirs: np.ndarray) -> np.ndarray:
    """Per-row quaternions (wxyz) rotating +Y onto each unit direction ``dirs [N,3]``."""
    a = np.array([0.0, 1.0, 0.0], np.float32)
    b = dirs.astype(np.float32)
    dot = np.clip(b[:, 1], -1.0, 1.0)
    axis = np.cross(np.broadcast_to(a, b.shape), b)
    axis_n = np.linalg.norm(axis, axis=1)
    angle = np.arccos(dot)
    q = np.zeros((b.shape[0], 4), np.float32)
    safe = axis_n > 1e-6
    ax = np.zeros_like(b)
    ax[safe] = axis[safe] / axis_n[safe, None]
    s = np.sin(angle / 2)
    q[:, 0] = np.cos(angle / 2)
    q[:, 1:] = ax * s[:, None]
    q[(~safe) & (dot > 0)] = np.array([1.0, 0.0, 0.0, 0.0], np.float32)
    q[(~safe) & (dot < 0)] = np.array([0.0, 1.0, 0.0, 0.0], np.float32)
    return q


class SkeletonScene:
    """Manages one viser batched-mesh handle per named track (e.g. gen/gt).

    ``scene`` is a viser scene handle — pass ``server.scene`` for a shared scene
    or ``client.scene`` for a per-client (per-browser) isolated scene.
    """

    def __init__(self, scene, edges: np.ndarray, track_colors: dict[str, np.ndarray],
                 mesh_colors: dict[str, np.ndarray] | None = None):
        self.scene = scene
        self.edges = np.asarray(edges, dtype=np.int64)
        self.colors = track_colors
        # Body-mesh colors default to the kimodo blue for every track.
        self.mesh_colors = mesh_colors or {name: MESH_COLOR for name in track_colors}
        self._bv, self._bf = bone_mesh()
        n = self.edges.shape[0]
        z_pos = np.zeros((n, 3), np.float32)
        z_wxyz = np.tile(np.array([1.0, 0, 0, 0], np.float32), (n, 1))
        z_scale = np.ones((n,), np.float32)
        self.handles = {}
        for name, color in track_colors.items():
            self.handles[name] = scene.add_batched_meshes_simple(
                f"/{name}_bones", self._bv, self._bf, z_wxyz.copy(), z_pos.copy(),
                batched_scales=z_scale.copy(), batched_colors=color, flat_shading=True,
            )
        self.mesh_handles: dict = {}      # name -> viser MeshHandle (body mesh)
        self.mesh_nfaces: dict = {}       # name -> face count (to detect LOD switch)

    def _transforms(self, joints: np.ndarray, f: int, offset: np.ndarray):
        ff = min(f, joints.shape[0] - 1)
        j = joints[ff] + offset
        p0 = j[self.edges[:, 0]]
        d = j[self.edges[:, 1]] - p0
        L = np.linalg.norm(d, axis=1)
        dirn = d / np.maximum(L, 1e-6)[:, None]
        return p0.astype(np.float32), quat_from_y(dirn), L.astype(np.float32)

    def update_track(self, name: str, joints, f: int, *, offset=None, show=True,
                     prefix_frames: int = 0) -> None:
        h = self.handles[name]
        h.visible = bool(show and joints is not None and joints.shape[0] > 0)
        if not h.visible:
            return
        off = np.zeros(3, np.float32) if offset is None else np.asarray(offset, np.float32)
        pos, wxyz, L = self._transforms(joints, f, off)
        h.batched_positions = pos
        h.batched_wxyzs = wxyz
        h.batched_scales = L
        try:
            h.batched_colors = self.colors[name] if f >= prefix_frames else PREFIX_COLOR
        except Exception:  # noqa: BLE001
            pass

    def update_mesh(self, name: str, verts, faces, f: int, *, offset=None, show=True) -> None:
        """Show the body mesh for a track at frame ``f`` (verts [T,V,3], faces [F,3]).

        Faces upload once; per frame only ``.vertices`` is reassigned (viser diffs
        it). A LOD switch (different face count) recreates the mesh node.
        """
        vis = bool(show and verts is not None and getattr(verts, "shape", (0,))[0] > 0
                   and faces is not None and faces.shape[0] > 0)
        if not vis:
            if name in self.mesh_handles:
                self.mesh_handles[name].visible = False
            return
        ff = min(f, verts.shape[0] - 1)
        vf = np.ascontiguousarray(verts[ff], dtype=np.float32)
        off = (0.0, 0.0, 0.0) if offset is None else tuple(float(x) for x in np.asarray(offset))
        if name not in self.mesh_handles or self.mesh_nfaces.get(name) != int(faces.shape[0]):
            if name in self.mesh_handles:
                try:
                    self.mesh_handles[name].remove()
                except Exception:  # noqa: BLE001
                    pass
            col = self.mesh_colors.get(name, MESH_COLOR)
            self.mesh_handles[name] = self.scene.add_mesh_simple(
                f"/{name}_mesh", vf, np.asarray(faces, np.int32), color=col,
                flat_shading=False, position=off,
            )
            self.mesh_nfaces[name] = int(faces.shape[0])
        else:
            h = self.mesh_handles[name]
            h.vertices = vf
            h.position = off
            h.visible = True

    def render_track(self, name: str, mode: str, f: int, *, joints=None, verts=None, faces=None,
                     offset=None, show=True, prefix_frames: int = 0) -> None:
        """Render one track as skeleton or body mesh per ``mode`` ('skeleton'|'mesh')."""
        if not show:
            self.handles[name].visible = False
            if name in self.mesh_handles:
                self.mesh_handles[name].visible = False
            return
        if mode == "mesh":
            self.handles[name].visible = False
            self.update_mesh(name, verts, faces, f, offset=offset, show=True)
        else:
            if name in self.mesh_handles:
                self.mesh_handles[name].visible = False
            self.update_track(name, joints, f, offset=offset, show=True, prefix_frames=prefix_frames)


# ---------------------------------------------------------------------------
# Text2Motion engine
# ---------------------------------------------------------------------------
class T2MEngine:
    """Trained t2m model + frozen Flan-T5 + frozen codec; text -> SOMA77 joints."""

    def __init__(self, model, text_enc, tok, edges, fps, device, fk_device):
        self.model = model
        self.text_enc = text_enc
        self.tok = tok
        self.edges = edges
        self.fps = float(fps)
        self.device = device
        self.fk_device = fk_device

    @classmethod
    def load(cls, checkpoint, *, tokenizer_checkpoint=None, text_encoder="flan",
             text_encoder_model=None, device="cuda:0"):
        import torch

        from semoco_generator.eval.rollout import load_model
        from semoco_generator.paths import default_checkpoint
        from semoco_generator.text import get_encoder_cls
        from semoco_generator.tokenizer_bridge import FrozenMotionTokenizer, soma_skeleton_edges

        dev = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"[t2m] loading model {checkpoint}", flush=True)
        model, _ = load_model(checkpoint, device=dev)
        if not model.cfg.use_text:
            raise SystemExit("checkpoint is not a text2motion model (use_text=False)")
        enc_cls = get_encoder_cls(text_encoder)
        model_id = text_encoder_model or enc_cls.DEFAULT_MODEL_ID
        print(f"[t2m] loading text encoder {text_encoder} -> {model_id}", flush=True)
        text_enc = enc_cls.load(model_id, device=dev)
        if text_enc.clip_dim != model.cfg.clip_dim:
            raise SystemExit(f"text clip_dim {text_enc.clip_dim} != model {model.cfg.clip_dim}")
        tok_ckpt = tokenizer_checkpoint or str(default_checkpoint())
        print(f"[t2m] loading tokenizer {tok_ckpt}", flush=True)
        tok = FrozenMotionTokenizer.load(tok_ckpt, device=dev)
        fk_device = "cuda" if dev.type == "cuda" else "cpu"
        return cls(model, text_enc, tok, soma_skeleton_edges(), tok.spec.source_fps, dev, fk_device)

    # -- anchors --
    @staticmethod
    def canonical_anchor():
        from semoco_generator.tokenizer_bridge import canonical_anchor
        return canonical_anchor(identity_dim=10)

    # -- decode --
    def decode_codes(self, codes: np.ndarray, anchor: dict) -> np.ndarray:
        """``codes [T_tok, Q]`` + anchor dict -> ``joints77 [T, 77, 3]`` (skeleton)."""
        if codes.shape[0] < 1:
            return np.zeros((0, 77, 3), np.float32)
        out = self.tok.decode_to_joints_arrays(
            codes.astype(np.int64),
            init_root_pos=anchor["init_root_pos"], init_root_rot6d=anchor["init_root_rot6d"],
            init_joints76_rot6d=anchor["init_joints76_rot6d"], identity_coeffs=anchor["identity_coeffs"],
            device=self.fk_device,
        )
        return out["joints77"].astype(np.float32)

    def decode_mesh(self, codes: np.ndarray, anchor: dict, *, low_lod: bool = True):
        """``codes [T_tok, Q]`` + anchor -> (``vertices [T,V,3]``, ``faces [F,3]``).

        ``low_lod`` True -> 4505 verts (fast, streamable), False -> 18056 verts.
        """
        if codes.shape[0] < 1:
            return np.zeros((0, 0, 3), np.float32), np.zeros((0, 3), np.int32)
        out = self.tok.decode_to_mesh_arrays(
            codes.astype(np.int64),
            init_root_pos=anchor["init_root_pos"], init_root_rot6d=anchor["init_root_rot6d"],
            init_joints76_rot6d=anchor["init_joints76_rot6d"], identity_coeffs=anchor["identity_coeffs"],
            low_lod=low_lod, device=self.fk_device,
        )
        return out["vertices"].astype(np.float32), out["faces"].astype(np.int32)

    def _encode_text(self, prompt: str):
        return self.text_enc.encode([prompt])   # (emb [1,L,dim], mask [1,L])

    # -- generation --
    def generate_codes(self, prompt: str, *, cfg_scale=3.0, temperature=0.9, max_tok=512) -> np.ndarray:
        """One-shot: prompt -> motion codes ``[T_tok, Q]`` (decode to joints/mesh separately)."""
        from semoco_generator.eval.rollout import SamplingConfig, generate_from_text

        emb, mask = self._encode_text(prompt)
        seqs = generate_from_text(
            self.model, emb, mask, max_tok=max_tok, cfg_scale=cfg_scale,
            sampling=SamplingConfig(temperature=temperature, top_p=0.9), device=self.device,
        )
        return seqs[0].cpu().numpy().astype(np.int64)

    def generate(self, prompt: str, *, cfg_scale=3.0, temperature=0.9, max_tok=512,
                 anchor: dict | None = None) -> np.ndarray:
        """One-shot: prompt -> joints77 [T,77,3]."""
        codes = self.generate_codes(prompt, cfg_scale=cfg_scale, temperature=temperature, max_tok=max_tok)
        return self.decode_codes(codes, anchor or self.canonical_anchor())

    def stream_generate(self, prompt: str, *, cfg_scale=3.0, temperature=0.9, max_tok=512,
                        chunk=4, anchor: dict | None = None):
        """Streaming skeleton: yields (joints77_so_far [T,77,3], committed_frames, n_tok, codes).

        Re-decodes the accumulated codes each chunk (cheap). ``committed_frames``
        is the count of already-stable frames (``T - STREAM_SETTLE_FRAMES``); the
        tail beyond it is provisional and refined on the next chunk. ``codes`` is
        the accumulated ``[T_tok, Q]`` so the caller can build the mesh afterward.
        """
        from semoco_generator.eval.rollout import SamplingConfig, generate_from_text_stream

        anchor = anchor or self.canonical_anchor()
        emb, mask = self._encode_text(prompt)
        for codes_list in generate_from_text_stream(
            self.model, emb, mask, max_tok=max_tok, cfg_scale=cfg_scale, chunk=chunk,
            sampling=SamplingConfig(temperature=temperature, top_p=0.9), device=self.device,
        ):
            codes = codes_list[0].cpu().numpy().astype(np.int64)
            n_tok = int(codes.shape[0])
            if n_tok < 1:
                continue
            joints = self.decode_codes(codes, anchor)
            committed = max(0, joints.shape[0] - STREAM_SETTLE_FRAMES)
            yield joints, committed, n_tok, codes
