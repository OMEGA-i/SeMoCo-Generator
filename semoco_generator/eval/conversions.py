"""Shared motion-representation conversion graph.

Models emit their *native* representation only. Tracks declare the
representation they *require*. This graph routes native -> required through a
small set of canonical relay representations instead of implementing an N x N
matrix of direct conversions:

    motion_codes --decode--> soma77 --index--> joints22 --process_file--> hml263
    smpl_rot6d_transl --smpl_fk--> joints22
    smpl_rot6d_transl --smpl_fk--> smpl_vertices --soma_fk--> soma77
    joints22 --smpl_fit--> smpl_vertices --soma_fk--> soma77

Shortest-path routing means, e.g., a SMPL-family model reaches ``hml263`` via
``joints22`` (never detouring through ``soma77``), while reaching ``soma77`` for
the TMR track goes through ``smpl_vertices``.

Path-finding (:func:`ConversionGraph.find_path`) is pure-python and testable
without torch; the edge implementations import heavy deps lazily.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .motion_batch import MotionBatch
from .schema import MotionClip, MotionRep
from .soma_pose import semoco_pose_aux


@dataclass
class ConversionContext:
    """Per-run resources the edges may need.

    Only the fields relevant to a given model's conversion path are used; e.g.
    ``semoco_tokenizer`` / ``semoco_anchor`` are only touched by the
    ``motion_codes -> soma77`` edge.
    """

    device: str = "cuda:0"
    fk_device: str = "cpu"
    semoco_tokenizer: Any | None = None
    semoco_anchor: dict[str, Any] | None = None
    # Optional per-clip anchors keyed by prompt_id. When set, _codes_to_soma77*
    # looks up the anchor for each clip (via clip.aux["prompt_id"]) instead of
    # using the global semoco_anchor. Clips whose prompt_id is not found fall
    # back to semoco_anchor (canonical).
    prompt_id_anchors: dict[str, dict[str, Any]] | None = None
    _fitter: Any | None = field(default=None, repr=False)
    _soma: Any | None = field(default=None, repr=False)
    _smpl_cache: dict = field(default_factory=dict, repr=False)

    def fitter(self):
        if self._fitter is None:
            from .motion_ops.utils.conversion import Joints22ToSMPLVertices

            self._fitter = Joints22ToSMPLVertices(device=self.device)
        return self._fitter

    def soma_converter(self):
        if self._soma is None:
            from .motion_ops.soma_converter import SOMAConverter

            self._soma = SOMAConverter(device=self.device)
        return self._soma

    def smpl(self, batch_size: int):
        import torch

        from .motion_ops.smpl_utils import create_smpl

        key = (int(batch_size), str(self.device))
        if key not in self._smpl_cache:
            dev = self.device if str(self.device) != "cpu" else "cpu"
            self._smpl_cache[key] = create_smpl(int(batch_size), str(dev))
        return self._smpl_cache[key]


# SMPL and HumanML3D both use left-before-right for paired limbs (e.g.
# joint 1 = L_Hip, joint 2 = R_Hip in both conventions).  Verified against
# FACE_JOINT_INDX in vendor/motion_process.py and smplx rest-pose output.
# No remapping is needed.
_SMPL24_TO_HML22 = np.array(
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    dtype=np.int64,
)

EdgeFn = Callable[[MotionClip, ConversionContext], MotionClip]
BatchEdgeFn = Callable[[list[MotionClip], ConversionContext], list[MotionClip]]

# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------
_MAX_FRAMES_PER_BATCH = 8192
"""Max total frames in one batched GPU forward pass.  Micro-batches split larger
groups so we don't OOM when stacking hundreds of clips for SMPL / SOMA."""


def _stack_clips(
    clips: list[MotionClip],
    *,
    aux_keys: tuple[str, ...] = (),
) -> tuple[np.ndarray, list[int], float, dict[str, np.ndarray]]:
    """Concatenate clip arrays along the time axis.

    Returns:
        ``(stacked_array, bounds, fps, {aux_key: stacked_aux})`` where
        ``bounds[i]`` is the starting frame index for clip ``i`` and
        ``bounds[-1]`` is the total frame count.
    """
    bounds = [0]
    arrays = []
    aux_collected: dict[str, list[np.ndarray]] = {k: [] for k in aux_keys}
    fps = float(clips[0].fps)
    for c in clips:
        arr = np.asarray(c.array, dtype=np.float32)
        arrays.append(arr)
        bounds.append(bounds[-1] + arr.shape[0])
        for k in aux_keys:
            aux_collected[k].append(np.asarray(c.aux[k], dtype=np.float32))
    stacked = np.concatenate(arrays, axis=0)
    aux_stacked = {k: np.concatenate(v, axis=0) for k, v in aux_collected.items()}
    return stacked, bounds, fps, aux_stacked


def _unstack_clips(
    stacked: np.ndarray,
    bounds: list[int],
    fps: float,
    rep: MotionRep,
    *,
    aux: dict[str, np.ndarray] | None = None,
) -> list[MotionClip]:
    """Inverse of :func:`_stack_clips`: split a stacked array back into clips."""
    out = []
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1]
        clip_aux = {}
        if aux is not None:
            clip_aux = {k: v[s:e] for k, v in aux.items()}
        out.append(MotionClip(rep=rep, array=stacked[s:e], fps=fps, aux=clip_aux))
    return out


# ---------------------------------------------------------------------------
# Edge implementations (heavy deps imported lazily)
# ---------------------------------------------------------------------------
def _codes_to_soma77(clip: MotionClip, ctx: ConversionContext) -> MotionClip:
    if ctx.semoco_tokenizer is None or ctx.semoco_anchor is None:
        raise RuntimeError("motion_codes -> soma77 needs ctx.semoco_tokenizer + semoco_anchor")
    codes = np.asarray(clip.array, dtype=np.int64)
    anchor = _resolve_anchor(clip, ctx)
    decoded = ctx.semoco_tokenizer.decode_to_full_pose_arrays(
        codes,
        init_root_pos=anchor["init_root_pos"],
        init_root_rot6d=anchor["init_root_rot6d"],
        init_joints76_rot6d=anchor["init_joints76_rot6d"],
        identity_coeffs=anchor["identity_coeffs"],
        device=ctx.fk_device,
    )
    return MotionClip(
        rep="soma77",
        array=np.asarray(decoded["joints77"], dtype=np.float32),
        fps=clip.fps,
        aux=semoco_pose_aux(decoded),
    )


def _soma77_to_joints22(clip: MotionClip, ctx: ConversionContext) -> MotionClip:
    from .tracks.smpl_hml.conversion import soma77_to_joints22

    j = soma77_to_joints22(np.asarray(clip.array, dtype=np.float32))
    return MotionClip(rep="joints22", array=j, fps=clip.fps)


def _joints22_to_hml263(clip: MotionClip, ctx: ConversionContext) -> MotionClip:
    from .metrics import resample_fps
    from .tracks.smpl_hml.conversion import joints22_to_hml263

    j = np.asarray(clip.array, dtype=np.float32)
    if abs(float(clip.fps) - 20.0) > 1e-3:
        j = resample_fps(j, float(clip.fps), 20.0)
    feats = joints22_to_hml263(j, 20.0)
    return MotionClip(rep="hml263", array=feats, fps=20.0)


def _joints22_to_smpl_vertices(clip: MotionClip, ctx: ConversionContext) -> MotionClip:
    verts = ctx.fitter().joints22_to_vertices(np.asarray(clip.array, dtype=np.float32))
    return MotionClip(
        rep="smpl_vertices",
        array=np.asarray(verts.detach().cpu().numpy(), dtype=np.float32),
        fps=clip.fps,
    )


def _smpl_vertices_to_soma77(clip: MotionClip, ctx: ConversionContext) -> MotionClip:
    import torch

    verts = torch.as_tensor(np.asarray(clip.array, dtype=np.float32))
    soma = ctx.soma_converter().vertices_to_soma77(verts)
    return MotionClip(rep="soma77", array=np.asarray(soma, dtype=np.float32), fps=clip.fps)


def _smpl_forward(clip: MotionClip, ctx: ConversionContext):
    """Run SMPL forward on ``smpl_rot6d_transl`` and return the SMPL output.

    Mirrors the HYMotion decode: 22 rot6d joints -> axis-angle, joint 0 is the
    global orient, joints 1..21 are body pose (SMPL expects 23 body slots, so
    the trailing 2 stay zero).
    """
    import torch

    from .motion_ops.utils.rot6d import rotation_6d_to_axis_angle

    rot6d = torch.as_tensor(np.asarray(clip.array, dtype=np.float32))
    transl = torch.as_tensor(np.asarray(clip.aux["transl"], dtype=np.float32))
    aa = rotation_6d_to_axis_angle(rot6d.to(torch.float32))
    go = aa[:, 0:1, :]
    bp = torch.zeros(aa.shape[0], 23, 3, dtype=aa.dtype)
    bp[:, :21, :] = aa[:, 1:22, :]
    T = int(rot6d.shape[0])
    smpl = ctx.smpl(T)
    dev = next(smpl.parameters()).device if hasattr(smpl, "parameters") else torch.device("cpu")
    with torch.no_grad():
        return smpl(
            global_orient=go.to(dev),
            body_pose=bp.to(dev),
            transl=transl.to(dev),
        )


def _smpl_rot6d_to_joints22(clip: MotionClip, ctx: ConversionContext) -> MotionClip:
    out = _smpl_forward(clip, ctx)
    j = out.joints[:, _SMPL24_TO_HML22, :].detach().cpu().numpy().astype(np.float32)
    return MotionClip(rep="joints22", array=j, fps=clip.fps)


def _smpl_rot6d_to_vertices(clip: MotionClip, ctx: ConversionContext) -> MotionClip:
    out = _smpl_forward(clip, ctx)
    v = out.vertices.detach().cpu().numpy().astype(np.float32)
    return MotionClip(rep="smpl_vertices", array=v, fps=clip.fps)


# ---------------------------------------------------------------------------
# Batch edge implementations
# ---------------------------------------------------------------------------
def _resolve_anchor(clip: MotionClip, ctx: ConversionContext) -> dict[str, Any]:
    """Return the per-clip anchor when available, otherwise the global canonical anchor.

    Annotates ``clip.aux["anchor_source"]`` with ``"per_clip"`` or ``"canonical"``
    so downstream diagnostics can distinguish which clips used GT anchors.
    """
    global_anchor: dict[str, Any] = ctx.semoco_anchor  # type: ignore[assignment]
    if ctx.prompt_id_anchors is None:
        _set_anchor_source(clip, "canonical")
        return global_anchor
    pid = (clip.aux or {}).get("prompt_id")
    if pid is None:
        _set_anchor_source(clip, "canonical")
        return global_anchor
    anchor = ctx.prompt_id_anchors.get(pid)
    if anchor is not None:
        _set_anchor_source(clip, "per_clip")
        return anchor
    _set_anchor_source(clip, "canonical")
    return global_anchor


def _set_anchor_source(clip: MotionClip, source: str) -> None:
    """Record the anchor source in clip aux metadata."""
    if clip.aux is None:
        clip.aux = {}
    clip.aux["anchor_source"] = source


def _codes_to_soma77_batch(clips: list[MotionClip], ctx: ConversionContext) -> list[MotionClip]:
    """Batch decode motion codes with complete SOMA rotations and translation.

    Groups codes by equal token length for exact zero-padding batch VQ decode,
    matching the GT path's approach.
    """
    if ctx.semoco_tokenizer is None or ctx.semoco_anchor is None:
        raise RuntimeError("motion_codes -> soma77 needs ctx.semoco_tokenizer + semoco_anchor")

    codes_list: list[np.ndarray] = []
    anchors_list: list[dict[str, np.ndarray]] = []
    identities_list: list[np.ndarray] = []
    for clip in clips:
        codes_list.append(np.asarray(clip.array, dtype=np.int64))
        a = _resolve_anchor(clip, ctx)
        anchors_list.append({
            "init_root_pos": a["init_root_pos"],
            "init_root_rot6d": a["init_root_rot6d"],
            "init_joints76_rot6d": a["init_joints76_rot6d"],
        })
        identities_list.append(np.asarray(a["identity_coeffs"], dtype=np.float32))

    decoded = ctx.semoco_tokenizer.decode_to_full_pose_arrays_batch(
        codes_list, anchors_list, identities_list,
        device=ctx.device,  # GPU FK — matches GT path
    )

    out: list[MotionClip] = []
    for d, clip in zip(decoded, clips):
        joints77 = np.asarray(d["joints77"], dtype=np.float32)
        out.append(
            MotionClip(
                rep="soma77",
                array=joints77,
                fps=clip.fps,
                aux=semoco_pose_aux(d),
            )
        )
    return out


def _joints22_to_smpl_vertices_batch(
    clips: list[MotionClip], ctx: ConversionContext
) -> list[MotionClip]:
    """Batch SMPL fitting: stack all frames, fit once, split back."""
    if not clips:
        return []
    stacked, bounds, fps, _aux = _stack_clips(clips)
    total_frames = bounds[-1]

    import torch as _torch

    if total_frames <= _MAX_FRAMES_PER_BATCH:
        verts = ctx.fitter().joints22_to_vertices(_torch.as_tensor(stacked))
        verts_np = np.asarray(verts.detach().cpu().numpy(), dtype=np.float32)
        return _unstack_clips(verts_np, bounds, fps, "smpl_vertices")

    # Micro-batch: process in chunks of _MAX_FRAMES_PER_BATCH frames
    all_verts = []
    for start in range(0, total_frames, _MAX_FRAMES_PER_BATCH):
        end = min(start + _MAX_FRAMES_PER_BATCH, total_frames)
        chunk = _torch.as_tensor(stacked[start:end])
        verts = ctx.fitter().joints22_to_vertices(chunk)
        all_verts.append(np.asarray(verts.detach().cpu().numpy(), dtype=np.float32))
    verts_np = np.concatenate(all_verts, axis=0)
    return _unstack_clips(verts_np, bounds, fps, "smpl_vertices")


def _smpl_vertices_to_soma77_batch(
    clips: list[MotionClip], ctx: ConversionContext
) -> list[MotionClip]:
    """Batch SOMA conversion: stack all frames, convert once, split back."""
    if not clips:
        return []
    import torch

    stacked, bounds, fps, _aux = _stack_clips(clips)
    total_frames = bounds[-1]

    if total_frames <= _MAX_FRAMES_PER_BATCH:
        verts_t = torch.as_tensor(stacked)
        soma = ctx.soma_converter().vertices_to_soma77(verts_t)
        return _unstack_clips(np.asarray(soma, dtype=np.float32), bounds, fps, "soma77")

    # Micro-batch
    all_soma = []
    for start in range(0, total_frames, _MAX_FRAMES_PER_BATCH):
        end = min(start + _MAX_FRAMES_PER_BATCH, total_frames)
        verts_t = torch.as_tensor(stacked[start:end])
        soma = ctx.soma_converter().vertices_to_soma77(verts_t)
        all_soma.append(np.asarray(soma, dtype=np.float32))
    return _unstack_clips(np.concatenate(all_soma, axis=0), bounds, fps, "soma77")


def _smpl_forward_batch(
    clips: list[MotionClip], ctx: ConversionContext
):
    """Run batched SMPL forward on ``smpl_rot6d_transl`` clips.

    Stacks all frames across clips, creates a single SMPL model, and returns
    the combined SMPL output.  Micro-batches if total frames > _MAX_FRAMES_PER_BATCH.
    """
    import torch

    from .motion_ops.utils.rot6d import rotation_6d_to_axis_angle

    stacked, bounds, fps, aux_stacked = _stack_clips(clips, aux_keys=("transl",))
    transl_stacked = aux_stacked["transl"]
    total_frames = bounds[-1]

    def _forward(chunk_rot6d, chunk_transl):
        rot6d_t = torch.as_tensor(chunk_rot6d)
        transl_t = torch.as_tensor(chunk_transl)
        aa = rotation_6d_to_axis_angle(rot6d_t.to(torch.float32))
        go = aa[:, 0:1, :]
        bp = torch.zeros(aa.shape[0], 23, 3, dtype=aa.dtype)
        bp[:, :21, :] = aa[:, 1:22, :]
        T = int(rot6d_t.shape[0])
        smpl = ctx.smpl(T)
        dev = next(smpl.parameters()).device if hasattr(smpl, "parameters") else torch.device("cpu")
        with torch.no_grad():
            return smpl(
                global_orient=go.to(dev),
                body_pose=bp.to(dev),
                transl=transl_t.to(dev),
            )

    if total_frames <= _MAX_FRAMES_PER_BATCH:
        out = _forward(stacked, transl_stacked)
        # Detach and move to CPU so callers always get numpy arrays.
        class _SMPLOut:
            pass
        result = _SMPLOut()
        result.joints = out.joints[:, _SMPL24_TO_HML22, :].detach().cpu().numpy().astype(np.float32)
        result.vertices = out.vertices.detach().cpu().numpy().astype(np.float32)
        return result, bounds, fps

    # Micro-batch and accumulate
    all_joints = []
    all_vertices = []
    for start in range(0, total_frames, _MAX_FRAMES_PER_BATCH):
        end = min(start + _MAX_FRAMES_PER_BATCH, total_frames)
        out = _forward(stacked[start:end], transl_stacked[start:end])
        all_joints.append(out.joints[:, _SMPL24_TO_HML22, :].detach().cpu().numpy().astype(np.float32))
        all_vertices.append(out.vertices.detach().cpu().numpy().astype(np.float32))
    # Return a synthetic object carrying the concatenated arrays
    class _SMPLOut:
        pass
    result = _SMPLOut()
    result.joints = np.concatenate(all_joints, axis=0)
    result.vertices = np.concatenate(all_vertices, axis=0)
    return result, bounds, fps


def _joints22_to_soma77_device_resident_batch(
    clips: list[MotionClip], ctx: ConversionContext
) -> list[MotionClip]:
    """Direct ``joints22 -> soma77`` edge that keeps the SMPL-vertices
    intermediate resident on ``ctx.device`` between the fit and SOMA-convert
    steps, instead of round-tripping through CPU numpy in between.

    The previous 2-hop path (``joints22 -> smpl_vertices`` then
    ``smpl_vertices -> soma77``) converts the fitted vertices to CPU numpy to
    build an intermediate :class:`MotionClip`, then re-stacks and re-uploads
    them for the SOMA step. Both GPU steps here operate on the same
    :class:`~semoco_generator.eval.motion_batch.MotionBatch` tensor without
    that intermediate CPU stop. Registering this as a direct edge also makes
    it the shortest-path choice for any model whose native/converted
    representation is ``joints22`` and target is ``soma77``.
    """
    if not clips:
        return []
    batch = MotionBatch.from_clips(clips, device=ctx.device)
    parts: list[np.ndarray] = []
    for chunk in batch.slice_by_frame_budget(_MAX_FRAMES_PER_BATCH):
        verts = ctx.fitter().joints22_to_vertices(chunk.data)  # GPU tensor, resident on ctx.device
        soma = ctx.soma_converter().vertices_to_soma77(verts)  # consumes the GPU tensor directly
        parts.append(np.asarray(soma, dtype=np.float32))
    combined = np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
    import torch as _torch

    result = MotionBatch(rep="soma77", data=_torch.as_tensor(combined), lengths=batch.lengths, fps=batch.fps)
    return result.split()


def _smpl_rot6d_to_joints22_batch(
    clips: list[MotionClip], ctx: ConversionContext
) -> list[MotionClip]:
    """Batch SMPL forward -> joints22."""
    if not clips:
        return []
    out, bounds, fps = _smpl_forward_batch(clips, ctx)
    return _unstack_clips(out.joints, bounds, fps, "joints22")


def _smpl_rot6d_to_vertices_batch(
    clips: list[MotionClip], ctx: ConversionContext
) -> list[MotionClip]:
    """Batch SMPL forward -> smpl_vertices."""
    if not clips:
        return []
    out, bounds, fps = _smpl_forward_batch(clips, ctx)
    return _unstack_clips(out.vertices, bounds, fps, "smpl_vertices")


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
class ConversionGraph:
    """Directed graph of motion-representation conversions with BFS routing.

    Supports both single-clip (:meth:`convert`) and batched (:meth:`convert_batch`)
    conversion.  Batch edges stack clips along the time axis for GPU-heavy steps
    (SMPL, SOMA, fitting), amortizing kernel-launch overhead.
    """

    def __init__(self) -> None:
        self._edges: dict[tuple[MotionRep, MotionRep], EdgeFn] = {}
        self._batch_edges: dict[tuple[MotionRep, MotionRep], BatchEdgeFn] = {}
        self._step_names: dict[tuple[MotionRep, MotionRep], str] = {}
        self._adj: dict[MotionRep, list[MotionRep]] = {}

    def register(self, src: MotionRep, dst: MotionRep, fn: EdgeFn, *, step: str) -> None:
        self._edges[(src, dst)] = fn
        self._step_names[(src, dst)] = step
        self._adj.setdefault(src, []).append(dst)
        self._adj.setdefault(dst, [])

    def register_batch(
        self, src: MotionRep, dst: MotionRep, fn: BatchEdgeFn, *, step: str
    ) -> None:
        """Register a batched edge.  Also registers a single-clip wrapper so
        :meth:`convert` continues to work for this edge."""
        self._batch_edges[(src, dst)] = fn
        single_fn: EdgeFn = lambda clip, ctx: fn([clip], ctx)[0]
        self._edges[(src, dst)] = single_fn
        self._step_names[(src, dst)] = step
        self._adj.setdefault(src, []).append(dst)
        self._adj.setdefault(dst, [])

    def find_path(self, src: MotionRep, dst: MotionRep) -> list[MotionRep] | None:
        """Shortest edge path from ``src`` to ``dst`` (inclusive), or None."""
        if src == dst:
            return [src]
        if src not in self._adj:
            return None
        prev: dict[MotionRep, MotionRep] = {src: src}
        q: deque[MotionRep] = deque([src])
        while q:
            node = q.popleft()
            for nxt in self._adj.get(node, ()):  # deterministic: insertion order
                if nxt in prev:
                    continue
                prev[nxt] = node
                if nxt == dst:
                    path = [dst]
                    while path[-1] != src:
                        path.append(prev[path[-1]])
                    return list(reversed(path))
                q.append(nxt)
        return None

    def convert(
        self,
        clip: MotionClip,
        target: MotionRep,
        ctx: ConversionContext,
    ) -> MotionClip:
        """Convert ``clip`` to ``target``."""
        path = self.find_path(clip.rep, target)
        if path is None:
            raise KeyError(f"no conversion path from {clip.rep!r} to {target!r}")
        cur = clip
        for src, dst in zip(path[:-1], path[1:]):
            fn = self._edges[(src, dst)]
            cur = fn(cur, ctx)
        return cur

    def convert_batch(
        self,
        clips: list[MotionClip],
        target: MotionRep,
        ctx: ConversionContext,
    ) -> list[MotionClip]:
        """Convert all *clips* to *target* using batched edges where available.

        All clips must share the same source representation.  GPU edges
        (SMPL forward, fitting, SOMA) are amortized across clips by stacking
        along the time axis.  Pure-numpy edges fall back to per-clip.
        """
        if not clips:
            return []

        src_rep = clips[0].rep
        for c in clips:
            if c.rep != src_rep:
                raise ValueError(
                    f"convert_batch requires uniform source rep, got {src_rep!r} and {c.rep!r}"
                )

        path = self.find_path(src_rep, target)
        if path is None:
            raise KeyError(f"no conversion path from {src_rep!r} to {target!r}")

        # Identity short-circuit
        if len(path) == 1:
            return list(clips)

        cur = list(clips)
        for src, dst in zip(path[:-1], path[1:]):
            batch_fn = self._batch_edges.get((src, dst))
            if batch_fn is not None:
                cur = batch_fn(cur, ctx)
            else:
                fn = self._edges[(src, dst)]
                cur = [fn(c, ctx) for c in cur]

        return cur


def build_default_graph() -> ConversionGraph:
    """The canonical dual-track conversion graph.

    GPU-heavy edges are registered with batch variants that stack clips along
    the time axis to amortize kernel-launch overhead (SMPL, SOMA, fitting).
    """
    g = ConversionGraph()
    # motion_codes -> soma77 (tokenizer decode; batch falls back to per-clip)
    g.register_batch("motion_codes", "soma77", _codes_to_soma77_batch, step="codes_to_soma77")
    # Pure-numpy edges (fast enough without batching)
    g.register("soma77", "joints22", _soma77_to_joints22, step="soma77_to_joints22")
    g.register("joints22", "hml263", _joints22_to_hml263, step="joints22_to_process_file_hml263")
    # GPU edges with batch variants
    g.register_batch("joints22", "smpl_vertices", _joints22_to_smpl_vertices_batch, step="joints22_to_smpl_vertices")
    g.register_batch("smpl_vertices", "soma77", _smpl_vertices_to_soma77_batch, step="smpl_vertices_to_soma77")
    # Direct device-resident shortcut: BFS prefers this 1-hop edge over the
    # 2-hop joints22->smpl_vertices->soma77 path, avoiding the intermediate
    # GPU->CPU->GPU round trip for joints22-native models on soma_tmr.
    g.register_batch(
        "joints22", "soma77", _joints22_to_soma77_device_resident_batch,
        step="joints22_to_soma77_device_resident",
    )
    # Register smpl_vertices before joints22 for this source so BFS still
    # prefers the dedicated smpl_rot6d_transl -> smpl_vertices -> soma77
    # route over detouring through the (lossier, joints-only) joints22
    # relay now that joints22 -> soma77 is a direct edge. Both are 2 hops;
    # adjacency-list order is the tie-breaker.
    g.register_batch("smpl_rot6d_transl", "smpl_vertices", _smpl_rot6d_to_vertices_batch, step="smpl_rot6d_to_smpl_vertices")
    g.register_batch("smpl_rot6d_transl", "joints22", _smpl_rot6d_to_joints22_batch, step="smpl_rot6d_to_joints22")
    return g


__all__ = [
    "ConversionContext",
    "ConversionGraph",
    "build_default_graph",
    "EdgeFn",
    "BatchEdgeFn",
]
