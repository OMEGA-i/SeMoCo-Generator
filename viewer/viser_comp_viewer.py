"""Offline gen-vs-gt comparison viewer for text2motion (test split).

For each test-split clip, generates motion from the clip's caption (gen, orange)
and renders it against the clip's ground-truth motion (gt, blue) in a viser 3D
viewport — a qualitative check of text2motion quality. Both are decoded with the
clip's GROUND-TRUTH frame-0 anchor so they start from the same pose and are
directly comparable.

Run::

    python viewer/viser_comp_viewer.py \
        --checkpoint runs/t2m_150m_flan/model/best.pt \
        --codes-root local://t2m_codes --split test \
        --tokenizer-checkpoint <codec.pt> --port 8080
    # then: ssh -N -L 8080:localhost:8080 <host> ; open http://localhost:8080
"""

from __future__ import annotations

import argparse
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
from viewer.t2m_engine import (  # noqa: E402
    GEN_COLOR, GT_COLOR, MESH_COLOR, MESH_GEN_COLOR, SkeletonScene, T2MEngine,
)


def _gt_anchor(ds, row_idx: int) -> dict:
    row = np.asarray(ds.anchors[row_idx], dtype=np.float32)
    return {
        "init_root_pos": row[0:3], "init_root_rot6d": row[3:9],
        "init_joints76_rot6d": row[9:465],
        "identity_coeffs": np.asarray(ds.identities[row_idx], dtype=np.float32),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Offline gen-vs-gt text2motion comparison viewer.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--codes-root", type=str, required=True)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--tokenizer-checkpoint", type=str, default=None)
    p.add_argument("--text-encoder", type=str, default="flan",
                   choices=["flan", "siglip", "qwen3"], help="registered text encoder key")
    p.add_argument("--text-encoder-model", type=str, default=None,
                   help="override model ID (defaults to built-in per key)")
    p.add_argument("--num-clips", type=int, default=200, help="how many clips to expose in the dropdown")
    p.add_argument("--max-motion-tok", type=int, default=300)
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--device", type=str, default="cuda:0")
    args = p.parse_args()

    from semoco_generator.dataset import T2MCodeDataset

    engine = T2MEngine.load(args.checkpoint, tokenizer_checkpoint=args.tokenizer_checkpoint,
                            text_encoder=args.text_encoder,
                            text_encoder_model=args.text_encoder_model, device=args.device)
    ds = T2MCodeDataset(args.codes_root, args.split, max_motion_tok=args.max_motion_tok)
    n = min(args.num_clips, len(ds))
    labels = []
    for i in range(n):
        e = ds.entry(i)
        cap = (e.get("caption") or "")[:40]
        labels.append(f"{i:04d} | {cap}")

    server = viser.ViserServer(host=args.host, port=args.port)
    for setup in (lambda: server.scene.set_up_direction("+y"),
                  lambda: server.scene.add_grid("/ground", plane="xz", infinite_grid=True,
                                                 cell_size=0.5, section_size=1.0, fade_distance=12.0)):
        try:
            setup()
        except Exception:  # noqa: BLE001
            pass

    # NOTE: comp viewer is single shared session (fine for one reviewer). If you
    # need multi-browser isolation like viser_live_viewer, move the GUI/scene into
    # server.on_client_connect and use client.gui/client.scene per the live viewer.
    scene = SkeletonScene(server.scene, engine.edges, {"gen": GEN_COLOR, "gt": GT_COLOR},
                          mesh_colors={"gen": MESH_GEN_COLOR, "gt": MESH_COLOR})

    gui_clip = server.gui.add_dropdown("Clip", options=labels, initial_value=labels[0])
    gui_caption = server.gui.add_text("Caption", initial_value="", disabled=True)
    gui_render = server.gui.add_dropdown(
        "Render", options=["Skeleton", "Mesh (low)", "Mesh (high)"], initial_value="Skeleton")
    gui_cfg = server.gui.add_slider("CFG scale", min=1.0, max=6.0, step=0.5, initial_value=3.0)
    gui_temp = server.gui.add_slider("Temperature", min=0.1, max=1.5, step=0.05, initial_value=0.9)
    gui_maxtok = server.gui.add_slider("Max tokens", min=10, max=2048, step=10, initial_value=512)
    gui_regen = server.gui.add_button("Regenerate")
    gui_show_gen = server.gui.add_checkbox("Show gen (orange)", initial_value=True)
    gui_show_gt = server.gui.add_checkbox("Show gt (blue)", initial_value=True)
    gui_offset = server.gui.add_checkbox("Side-by-side", initial_value=True)
    gui_play = server.gui.add_button("Play / Pause")
    gui_speed = server.gui.add_slider("Speed", min=0.1, max=3.0, step=0.1, initial_value=1.0)
    gui_frame = server.gui.add_slider("Frame", min=0, max=1, step=1, initial_value=0)
    gui_status = server.gui.add_text("Status", initial_value="", disabled=True)

    # mesh: {(who, low_lod): (verts, faces)} for the current clip. gen_codes cached per settings.
    state = {"gen_j": None, "gt_j": None, "gen_codes": None, "gt_codes": None, "anchor": None,
             "T": 1, "playing": False, "busy": False, "code_cache": {}, "mesh": {}, "_last": None}

    def _render(f: int) -> None:
        mode = gui_render.value
        # De-dup repeat renders of the same view (play loop sets the slider AND
        # calls _render) to avoid double-sending the heavy mesh vertex buffer.
        key = (f, mode, gui_show_gen.value, gui_show_gt.value, gui_offset.value)
        if state["_last"] == key:
            return
        state["_last"] = key
        off = np.array([1.2, 0.0, 0.0], np.float32) if gui_offset.value else np.zeros(3, np.float32)
        with server.atomic():
            for who, track_off, show in (("gen", np.zeros(3, np.float32), gui_show_gen.value),
                                         ("gt", off, gui_show_gt.value)):
                if mode == "Skeleton":
                    scene.render_track(who, "skeleton", f, joints=state[f"{who}_j"], offset=track_off, show=show)
                else:
                    mv = state["mesh"].get((who, mode == "Mesh (low)"))
                    if mv is None:  # mesh not built yet -> skeleton fallback
                        scene.render_track(who, "skeleton", f, joints=state[f"{who}_j"], offset=track_off, show=show)
                    else:
                        scene.render_track(who, "mesh", f, verts=mv[0], faces=mv[1], offset=track_off, show=show)
        tg = 0 if state["gen_j"] is None else state["gen_j"].shape[0]
        tt = 0 if state["gt_j"] is None else state["gt_j"].shape[0]
        gui_status.value = f"frame {f}/{state['T']}  ({f / engine.fps:.1f}s)  gen {tg}f / gt {tt}f  [{mode}]"

    def _build_mesh(low: bool) -> None:
        if state["gen_codes"] is None or (("gen", low) in state["mesh"]):
            return
        gui_status.value = f"building {'low' if low else 'high'}-LOD mesh…"
        gv, gf = engine.decode_mesh(state["gen_codes"], state["anchor"], low_lod=low)
        tv, tf = engine.decode_mesh(state["gt_codes"], state["anchor"], low_lod=low)
        state["mesh"][("gen", low)] = (gv, gf)
        state["mesh"][("gt", low)] = (tv, tf)
        gui_status.value = f"mesh ready ({'low' if low else 'high'} LOD)"
        _render(int(gui_frame.value))

    def _maybe_build() -> None:
        mode = gui_render.value
        if mode.startswith("Mesh") and state["gen_codes"] is not None and not state["busy"]:
            _build_mesh(mode == "Mesh (low)")

    def _load(idx: int, regen: bool = False) -> None:
        if state["busy"]:
            return
        state["busy"] = True
        state["playing"] = False
        e = ds.entry(idx)
        caption = e.get("caption") or ""
        gui_caption.value = caption
        gui_status.value = f"generating: {caption!r} ..."
        try:
            anchor = _gt_anchor(ds, e["row"])
            gt_codes = np.asarray(
                ds.codes[e["code_start"]: e["code_start"] + e["code_len"]], dtype=np.int64)
            gt_j = engine.decode_codes(gt_codes, anchor)
            key = (idx, round(float(gui_cfg.value), 2), round(float(gui_temp.value), 2), int(gui_maxtok.value))
            if regen or key not in state["code_cache"]:
                state["code_cache"][key] = engine.generate_codes(
                    caption, cfg_scale=float(gui_cfg.value), temperature=float(gui_temp.value),
                    max_tok=int(gui_maxtok.value))
            gen_codes = state["code_cache"][key]
            gen_j = engine.decode_codes(gen_codes, anchor)
        except Exception as exc:  # noqa: BLE001
            gui_status.value = f"failed: {exc}"
            state["busy"] = False
            return
        state.update({"gen_j": gen_j, "gt_j": gt_j, "gen_codes": gen_codes, "gt_codes": gt_codes,
                      "anchor": anchor, "mesh": {}})
        state["T"] = max(1, gen_j.shape[0], gt_j.shape[0])
        try:
            gui_frame.max = state["T"] - 1
        except Exception:  # noqa: BLE001
            pass
        gui_frame.value = 0
        _render(0)
        state["busy"] = False
        state["playing"] = True
        _maybe_build()

    @gui_clip.on_update
    def _(_=None) -> None:
        threading.Thread(target=_load, args=(gui_clip.options.index(gui_clip.value),), daemon=True).start()

    @gui_regen.on_click
    def _(_=None) -> None:
        threading.Thread(target=_load, args=(gui_clip.options.index(gui_clip.value), True), daemon=True).start()

    @gui_render.on_update
    def _(_=None) -> None:
        threading.Thread(target=_maybe_build, daemon=True).start()
        _render(int(gui_frame.value))

    @gui_frame.on_update
    def _(_=None) -> None:
        _render(int(gui_frame.value))

    for g in (gui_show_gen, gui_show_gt, gui_offset):
        g.on_update(lambda _=None: _render(int(gui_frame.value)))

    @gui_play.on_click
    def _(_=None) -> None:
        state["playing"] = not state["playing"]

    # Large mesh vertex buffers stream at ~25 fps (frame stride); skeleton at full fps.
    mesh_stride = max(1, round(engine.fps / 25.0))

    def play_loop() -> None:
        while True:
            if state["playing"] and not state["busy"] and state["T"] > 1:
                stride = mesh_stride if gui_render.value != "Skeleton" else 1
                nxt = (int(gui_frame.value) + stride) % state["T"]
                gui_frame.value = nxt
                _render(nxt)
                time.sleep(stride / max(1e-3, engine.fps * gui_speed.value))
            else:
                time.sleep(0.05)

    threading.Thread(target=play_loop, daemon=True).start()
    threading.Thread(target=_load, args=(0,), daemon=True).start()
    print(f"[comp] {n} test clips from {args.codes_root}; open the URL above", flush=True)
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
