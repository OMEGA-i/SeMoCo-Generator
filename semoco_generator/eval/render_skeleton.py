"""Render rollout SOMA77 skeletons to GIF/MP4 (pred vs gt, side by side).

Consumes the ``.npz`` written by ``rollout_eval`` / ``visualize_soma77``
(``joints77 [T,77,3]``, ``edges``, ``fps``, ``prefix_frames``) and produces an
animation with the predicted motion next to ground truth. Prefix frames
(the conditioning 2s) are drawn muted; the generated rollout is highlighted, so
the prefix -> rollout handoff is visible.

Example::

    python -m semoco_generator.eval.render_skeleton \
        --eval-dir runs/mgpt_codear_150m/eval_test --num 5 \
        --format mp4 --out-dir runs/mgpt_codear_150m/eval_test/render
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..local_uri import resolve_local_uri

_PREFIX_COLOR = "#9aa0a6"   # muted gray = conditioning prefix
_PRED_COLOR = "#ff7043"     # orange = generated rollout
_GT_COLOR = "#42a5f5"       # blue = ground truth rollout


def _bone_tris(joints_frame: np.ndarray, edges: np.ndarray, knuckle: float = 0.12, width: float = 0.09):
    """Octahedral 'bone' triangles for each edge -> [E*8, 3, 3] (Blender-style)."""
    p0 = joints_frame[edges[:, 0]]                      # [E,3]
    p1 = joints_frame[edges[:, 1]]
    d = p1 - p0
    L = np.linalg.norm(d, axis=1, keepdims=True)
    dirn = d / np.maximum(L, 1e-6)
    a = np.where(np.abs(dirn[:, 1:2]) < 0.9, np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    u = np.cross(a, dirn); u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-6)
    v = np.cross(dirn, u)
    mid = p0 + dirn * (knuckle * L)
    w = width * L
    r = [mid + u * w, mid + v * w, mid - u * w, mid - v * w]   # 4 ring pts, each [E,3]
    tris = []
    for i in range(4):
        j = (i + 1) % 4
        tris.append(np.stack([p0, r[i], r[j]], axis=1))        # base cap [E,3,3]
        tris.append(np.stack([p1, r[j], r[i]], axis=1))        # tip cap
    return np.concatenate(tris, axis=0)                        # [E*8, 3, 3]


def _load_track(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as d:
        out = {"joints77": np.asarray(d["joints77"], dtype=np.float32)}
        out["edges"] = np.asarray(d["edges"], dtype=np.int64) if "edges" in d else None
        out["fps"] = float(d["fps"]) if "fps" in d else 50.0
        out["prefix_frames"] = int(d["prefix_frames"]) if "prefix_frames" in d else 0
    return out


def _equal_bounds(*joint_arrays: np.ndarray, margin: float = 0.2):
    allj = np.concatenate([j.reshape(-1, 3) for j in joint_arrays], axis=0)
    lo = allj.min(axis=0)
    hi = allj.max(axis=0)
    center = (lo + hi) / 2
    span = float((hi - lo).max()) * (1.0 + margin)
    span = max(span, 1.0)
    return center, span


def render_clip(
    rec_id: str,
    eval_dir: Path,
    out_path: Path,
    *,
    fmt: str = "gif",
    stride: int = 2,
    max_frames: int = 400,
    rollout_color: str = _PRED_COLOR,
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    pred = _load_track(eval_dir / f"{rec_id}.pred.npz")
    gt_path = eval_dir / f"{rec_id}.gt.npz"
    gt = _load_track(gt_path) if gt_path.is_file() else None
    edges = pred["edges"]
    fps = pred["fps"]
    prefix_frames = pred["prefix_frames"]

    pj = pred["joints77"]                          # [T, 77, 3]
    gj = gt["joints77"] if gt is not None else None
    T = pj.shape[0]
    idx = list(range(0, T, max(1, stride)))
    if len(idx) > max_frames:
        idx = idx[:max_frames]

    # World axes -> plot axes: vertical is UMR y (axis 1); use Z-up plot.
    def to_plot(j):  # [...,3] (x,y,z) -> (x, z, y)
        return np.stack([j[..., 0], j[..., 2], j[..., 1]], axis=-1)

    pj_p = to_plot(pj)
    gj_p = to_plot(gj) if gj is not None else None
    bounds_src = [pj_p] + ([gj_p] if gj_p is not None else [])
    center, span = _equal_bounds(*bounds_src)

    ncols = 2 if gj_p is not None else 1
    fig = plt.figure(figsize=(6 * ncols, 6))
    panels = []
    specs = [("Pred (rollout)", pj_p, rollout_color)]
    if gj_p is not None:
        specs.append(("Ground truth", gj_p, _GT_COLOR))

    for col, (title, jp, color) in enumerate(specs):
        ax = fig.add_subplot(1, ncols, col + 1, projection="3d")
        ax.set_title(title)
        for axis, c in zip(("set_xlim", "set_ylim", "set_zlim"), center):
            getattr(ax, axis)(c - span / 2, c + span / 2)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.view_init(elev=12, azim=-60)
        # Ground grid at vertical=0 (plot Z; floor_y canonicalization).
        gx = np.linspace(center[0] - span / 2, center[0] + span / 2, 11)
        gy = np.linspace(center[1] - span / 2, center[1] + span / 2, 11)
        for x in gx:
            ax.plot([x, x], [gy[0], gy[-1]], [0, 0], color="#dddddd", linewidth=0.6, zorder=0)
        for y in gy:
            ax.plot([gx[0], gx[-1]], [y, y], [0, 0], color="#dddddd", linewidth=0.6, zorder=0)
        coll = Poly3DCollection(_bone_tris(jp[0], edges), facecolors=_PREFIX_COLOR, edgecolors="none")
        ax.add_collection3d(coll)
        panels.append((ax, coll, jp, color))

    txt = fig.text(0.5, 0.96, "", ha="center", fontsize=12)
    fps_out = max(1, int(round(fps / max(1, stride))))

    def update(fi):
        f = idx[fi]
        is_rollout = f >= prefix_frames
        phase = "ROLLOUT" if is_rollout else "PREFIX"
        txt.set_text(f"{rec_id}   frame {f}/{T}   {phase}   ({f / fps:.1f}s)")
        artists = [txt]
        for ax, coll, jp, color in panels:
            ff = min(f, jp.shape[0] - 1)   # gt may be shorter than pred
            c = color if is_rollout else _PREFIX_COLOR
            coll.set_verts(_bone_tris(jp[ff], edges))
            coll.set_facecolor(c)
            artists += [coll]
        return artists

    anim = FuncAnimation(fig, update, frames=len(idx), interval=1000 / fps_out, blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "mp4":
        # Point matplotlib at the bundled imageio-ffmpeg binary (no system ffmpeg).
        try:
            import imageio_ffmpeg
            matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:  # noqa: BLE001
            pass
        anim.save(str(out_path), writer=FFMpegWriter(fps=fps_out, bitrate=2400))
    else:
        anim.save(str(out_path), writer=PillowWriter(fps=fps_out))
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render rollout skeletons to GIF/MP4.")
    parser.add_argument("--eval-dir", type=str, required=True, help="dir with <rec>.pred.npz/.gt.npz (local:// ok)")
    parser.add_argument("--rec-id", type=str, default=None, help="single recording; else uses --num from index.json")
    parser.add_argument("--num", type=int, default=3, help="render first N recordings from index.json")
    parser.add_argument("--out-dir", type=str, default=None, help="default <eval-dir>/render")
    parser.add_argument("--format", choices=["gif", "mp4"], default="mp4")
    parser.add_argument("--stride", type=int, default=2, help="frame stride (50fps/stride playback)")
    parser.add_argument("--max-frames", type=int, default=400)
    args = parser.parse_args()

    eval_dir = resolve_local_uri(args.eval_dir)
    out_dir = resolve_local_uri(args.out_dir) if args.out_dir else eval_dir / "render"

    if args.rec_id:
        rec_ids = [args.rec_id]
    else:
        index = json.loads((eval_dir / "index.json").read_text())
        rec_ids = index["rec_ids"][: args.num]

    for rec_id in rec_ids:
        out_path = out_dir / f"{rec_id}.{args.format}"
        render_clip(rec_id, eval_dir, out_path, fmt=args.format, stride=args.stride, max_frames=args.max_frames)
        print(f"[render] {out_path}")


if __name__ == "__main__":
    main()
