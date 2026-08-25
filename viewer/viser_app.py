"""Viser-based 3D overlay player for SeMoCo-Generator rollouts.

Renders the predicted motion and ground truth as overlaid SOMA77 skeletons
in a real-time 3D viewport, reading the per-clip ``.npz`` from ``rollout_eval`` /
``visualize_soma77`` (``joints77``, ``edges``, ``fps``, ``prefix_frames``). The
conditioning prefix is drawn muted; the generated rollout is highlighted.

For text2motion, use the dedicated viewers instead:
  * ``viser_comp_viewer.py`` — offline gen-vs-gt comparison over the test split
  * ``viser_live_viewer.py`` — live streaming generation from a text prompt

Run::

    python viewer/viser_app.py --eval-dir runs/mgpt_150m/eval_test --port 8080
    python viewer/viser_app.py --runs-root runs --port 8080   # switch runs in GUI
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import viser  # noqa: E402

from semoco_generator.local_uri import resolve_local_uri  # noqa: E402
from viewer.t2m_engine import GEN_COLOR, GT_COLOR, SkeletonScene  # noqa: E402


def _load_npz(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as d:
        return {
            "joints": np.asarray(d["joints77"], dtype=np.float32),
            "edges": np.asarray(d["edges"], dtype=np.int64) if "edges" in d else None,
            "fps": float(d["fps"]) if "fps" in d else 50.0,
            "prefix_frames": int(d["prefix_frames"]) if "prefix_frames" in d else 0,
        }


class Clip:
    def __init__(self, eval_dir: Path, rec_id: str) -> None:
        self.rec_id = rec_id
        self.pred = _load_npz(eval_dir / f"{rec_id}.pred.npz")
        self.gt = _load_npz(eval_dir / f"{rec_id}.gt.npz")
        ref = self.pred or self.gt
        if ref is None:
            raise FileNotFoundError(f"no npz for {rec_id}")
        self.edges = ref["edges"]
        self.fps = ref["fps"]
        self.prefix_frames = ref["prefix_frames"]
        self.T = int(self.pred["joints"].shape[0]) if self.pred else int(self.gt["joints"].shape[0])


def _list_recordings(eval_dir: Path) -> list[str]:
    index = eval_dir / "index.json"
    if index.is_file():
        rec_ids = json.loads(index.read_text()).get("rec_ids", [])
        if rec_ids:
            return rec_ids
    return sorted(p.name[: -len(".pred.npz")] for p in eval_dir.glob("*.pred.npz"))


def _discover_eval_dirs(runs_root: Path, initial_eval_dir: Path | None = None) -> dict[str, Path]:
    """Return GUI labels -> eval directories for every populated ``*/eval_test``."""
    out: dict[str, Path] = {}
    if runs_root.is_dir():
        for p in sorted(runs_root.glob("*/eval_test")):
            if not p.is_dir():
                continue
            try:
                if _list_recordings(p):
                    out[p.parent.name] = p
            except Exception:  # noqa: BLE001
                continue
    if initial_eval_dir is not None:
        label = initial_eval_dir.parent.name if initial_eval_dir.name == "eval_test" else str(initial_eval_dir)
        out.setdefault(label, initial_eval_dir)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Viser 3D overlay player for SeMoCo-Generator rollouts.")
    parser.add_argument("--eval-dir", type=str, default=None, help="initial dir with <rec>.pred/.gt.npz (local:// ok)")
    parser.add_argument("--runs-root", type=str, default="runs", help="scan <runs-root>/*/eval_test for GUI run selection")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    initial_eval_dir = resolve_local_uri(args.eval_dir) if args.eval_dir else None
    runs_root = resolve_local_uri(args.runs_root)
    run_dirs = _discover_eval_dirs(runs_root, initial_eval_dir)
    if not run_dirs:
        raise SystemExit(f"no eval dirs found under {runs_root}/*/eval_test")

    run_labels = list(run_dirs)
    initial_run = (
        initial_eval_dir.parent.name
        if initial_eval_dir is not None and initial_eval_dir.name == "eval_test"
        else run_labels[0]
    )
    if initial_run not in run_dirs:
        initial_run = run_labels[0]
    eval_dir = run_dirs[initial_run]
    rec_ids = _list_recordings(eval_dir)

    server = viser.ViserServer(host=args.host, port=args.port)
    for setup in (lambda: server.scene.set_up_direction("+y"),
                  lambda: server.scene.add_grid("/ground", plane="xz", infinite_grid=True,
                                                 cell_size=0.5, section_size=1.0, fade_distance=12.0)):
        try:
            setup()
        except Exception:  # noqa: BLE001
            pass

    # ---- GUI ----
    gui_run = server.gui.add_dropdown("Run", options=run_labels, initial_value=initial_run)
    gui_refresh = server.gui.add_button("Refresh runs")
    gui_rec = server.gui.add_dropdown("Recording", options=rec_ids, initial_value=rec_ids[0])
    gui_show_pred = server.gui.add_checkbox("Show pred", initial_value=True)
    gui_show_gt = server.gui.add_checkbox("Show gt", initial_value=True)
    gui_offset = server.gui.add_checkbox("Side-by-side (offset gt)", initial_value=False)
    gui_play = server.gui.add_button("Play / Pause")
    gui_speed = server.gui.add_slider("Speed", min=0.1, max=3.0, step=0.1, initial_value=1.0)
    gui_frame = server.gui.add_slider("Frame", min=0, max=1, step=1, initial_value=0)
    gui_phase = server.gui.add_text("Phase", initial_value="", disabled=True)

    state = {"clip": None, "playing": False, "scene": None, "eval_dir": eval_dir, "run_dirs": run_dirs}

    def render(f: int) -> None:
        clip = state["clip"]
        scene = state["scene"]
        if clip is None or scene is None:
            return
        offset = np.array([1.2, 0.0, 0.0], np.float32) if gui_offset.value else np.zeros(3, np.float32)
        with server.atomic():
            scene.update_track("pred", clip.pred["joints"] if clip.pred else None, f,
                               offset=np.zeros(3, np.float32),
                               show=bool(clip.pred) and gui_show_pred.value,
                               prefix_frames=clip.prefix_frames)
            scene.update_track("gt", clip.gt["joints"] if clip.gt else None, f,
                               offset=offset, show=bool(clip.gt) and gui_show_gt.value,
                               prefix_frames=clip.prefix_frames)
        phase = "ROLLOUT" if f >= clip.prefix_frames else "PREFIX"
        gui_phase.value = f"{phase}  frame {f}/{clip.T}  ({f / clip.fps:.1f}s)"

    def load_clip(rec_id: str) -> None:
        clip = Clip(state["eval_dir"], rec_id)
        state["clip"] = clip
        if state["scene"] is None:
            state["scene"] = SkeletonScene(server.scene, clip.edges, {"pred": GEN_COLOR, "gt": GT_COLOR})
        try:
            gui_frame.max = max(1, clip.T - 1)
        except Exception:  # noqa: BLE001
            pass
        gui_frame.value = 0
        render(0)

    def load_run(label: str) -> None:
        state["eval_dir"] = state["run_dirs"][label]
        rec_ids_now = _list_recordings(state["eval_dir"])
        if not rec_ids_now:
            gui_phase.value = f"No recordings under {state['eval_dir']}"
            return
        gui_rec.options = rec_ids_now
        gui_rec.value = rec_ids_now[0]
        state["playing"] = False
        load_clip(rec_ids_now[0])

    @gui_rec.on_update
    def _(_=None) -> None:
        load_clip(gui_rec.value)

    @gui_run.on_update
    def _(_=None) -> None:
        load_run(gui_run.value)

    @gui_refresh.on_click
    def _(_=None) -> None:
        refreshed = _discover_eval_dirs(runs_root, initial_eval_dir)
        if not refreshed:
            gui_phase.value = f"No eval dirs found under {runs_root}/*/eval_test"
            return
        state["run_dirs"] = refreshed
        labels = list(refreshed)
        previous = gui_run.value if gui_run.value in refreshed else labels[0]
        gui_run.options = labels
        gui_run.value = previous
        load_run(previous)

    @gui_frame.on_update
    def _(_=None) -> None:
        render(int(gui_frame.value))

    for g in (gui_show_pred, gui_show_gt, gui_offset):
        g.on_update(lambda _=None: render(int(gui_frame.value)))

    @gui_play.on_click
    def _(_=None) -> None:
        state["playing"] = not state["playing"]

    def play_loop() -> None:
        while True:
            if state["playing"] and state["clip"] is not None:
                clip = state["clip"]
                nxt = (int(gui_frame.value) + 1) % clip.T
                gui_frame.value = nxt
                render(nxt)
                time.sleep(1.0 / max(1e-3, clip.fps * gui_speed.value))
            else:
                time.sleep(0.05)

    threading.Thread(target=play_loop, daemon=True).start()
    load_clip(rec_ids[0])
    print(f"[viser] {len(rec_ids)} recordings from {eval_dir}; {len(run_dirs)} run(s); open the URL above", flush=True)
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
