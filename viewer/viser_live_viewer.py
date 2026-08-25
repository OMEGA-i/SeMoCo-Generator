"""Live streaming text2motion viewer (per-client sessions).

Hosts a trained t2m checkpoint. Each connected browser gets its **own** GUI +
3D scene (built in ``on_client_connect`` on ``client.gui`` / ``client.scene``),
so multiple people can use it independently without their prompts/motions
interfering. You type a prompt, click Generate, and the motion is produced
autoregressively and **streamed to your view as it is generated** (the codec
re-decodes the accumulated tokens each chunk and the skeleton animates toward the
generation frontier — low time-to-first-motion). Streaming is over viser's own
websocket. The model is shared across clients; a global lock serialises the
actual generation so concurrent clients don't exhaust GPU memory.

Run::

    python viewer/viser_live_viewer.py \
        --checkpoint runs/t2m_150m/model/best.pt \
        --tokenizer-checkpoint <codec.pt> --port 8081 --chunk 4
    # then: ssh -N -L 8081:localhost:8081 <host> ; open http://localhost:8081
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

from viewer.t2m_engine import GEN_COLOR, SkeletonScene, T2MEngine  # noqa: E402

# One generation at a time across all clients (shared model + bounded GPU memory).
_GEN_LOCK = threading.Lock()


def main() -> None:
    p = argparse.ArgumentParser(description="Live streaming text2motion viewer (per-client).")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--tokenizer-checkpoint", type=str, default=None)
    p.add_argument("--text-encoder", type=str, default="flan",
                   choices=["flan", "siglip", "qwen3"], help="registered text encoder key")
    p.add_argument("--text-encoder-model", type=str, default=None,
                   help="override model ID (defaults to built-in per key)")
    p.add_argument("--chunk", type=int, default=4, help="packets per stream update (~chunk/12.5 s)")
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8081)
    p.add_argument("--device", type=str, default="cuda:0")
    args = p.parse_args()

    engine = T2MEngine.load(args.checkpoint, tokenizer_checkpoint=args.tokenizer_checkpoint,
                            text_encoder=args.text_encoder,
                            text_encoder_model=args.text_encoder_model, device=args.device)

    server = viser.ViserServer(host=args.host, port=args.port)

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        # Everything below is PER-CLIENT: its own GUI widgets, scene handles, and
        # generation state — isolated from other browsers.
        for setup in (lambda: client.scene.set_up_direction("+y"),
                      lambda: client.scene.add_grid("/ground", plane="xz", infinite_grid=True,
                                                    cell_size=0.5, section_size=1.0, fade_distance=12.0)):
            try:
                setup()
            except Exception:  # noqa: BLE001
                pass
        scene = SkeletonScene(client.scene, engine.edges, {"gen": GEN_COLOR})

        gui_prompt = client.gui.add_text("Prompt", initial_value="a person walks forward")
        gui_render = client.gui.add_dropdown(
            "Render", options=["Skeleton", "Mesh (low)", "Mesh (high)"], initial_value="Skeleton")
        gui_cfg = client.gui.add_slider("CFG scale", min=1.0, max=6.0, step=0.5, initial_value=3.0)
        gui_temp = client.gui.add_slider("Temperature", min=0.1, max=1.5, step=0.05, initial_value=0.9)
        gui_maxtok = client.gui.add_slider("Max tokens", min=10, max=2048, step=10, initial_value=512)
        gui_chunk = client.gui.add_slider("Stream chunk (tok)", min=1, max=32, step=1, initial_value=args.chunk)
        gui_generate = client.gui.add_button("Generate")
        gui_play = client.gui.add_button("Play / Pause")
        gui_speed = client.gui.add_slider("Speed", min=0.1, max=3.0, step=0.1, initial_value=1.0)
        gui_frame = client.gui.add_slider("Frame", min=0, max=1, step=1, initial_value=0)
        gui_status = client.gui.add_text("Status", initial_value="ready", disabled=True)

        # mesh: {low_lod(bool): (verts, faces)}. Built lazily after generation.
        st = {"joints": None, "T": 1, "playing": False, "busy": False, "generating": False,
              "codes": None, "anchor": None, "mesh": {}, "_last": None}

        def _render(f: int) -> None:
            mode = gui_render.value
            # De-dup: skip a repeat render of the same (frame, mode) during playback
            # (the play loop sets gui_frame.value AND calls _render -> avoid double
            # sending the heavy mesh vertex buffer). Never skip mid-generation.
            key = (f, mode)
            if not st["generating"] and st["_last"] == key:
                return
            st["_last"] = key
            with client.atomic():
                if mode == "Skeleton" or st["generating"]:
                    scene.render_track("gen", "skeleton", f, joints=st["joints"], show=st["joints"] is not None)
                else:
                    low = mode == "Mesh (low)"
                    mv = st["mesh"].get(low)
                    if mv is None:  # mesh not built yet -> fall back to skeleton
                        scene.render_track("gen", "skeleton", f, joints=st["joints"], show=st["joints"] is not None)
                    else:
                        scene.render_track("gen", "mesh", f, verts=mv[0], faces=mv[1], show=True)

        def _set_buffer(joints: np.ndarray) -> None:
            st["joints"] = joints
            st["T"] = max(1, joints.shape[0])
            try:
                gui_frame.max = st["T"] - 1
            except Exception:  # noqa: BLE001
                pass

        def _build_mesh(low: bool) -> None:
            """Build the full-clip body mesh for the current codes (cached per LOD)."""
            if st["codes"] is None or st["codes"].shape[0] < 1 or low in st["mesh"]:
                return
            with _GEN_LOCK:
                gui_status.value = f"building {'low' if low else 'high'}-LOD mesh…"
                verts, faces = engine.decode_mesh(st["codes"], st["anchor"], low_lod=low)
                st["mesh"][low] = (verts, faces)
                gui_status.value = f"mesh ready: {verts.shape[1]} verts / {faces.shape[0]} faces"
            _render(int(gui_frame.value))

        def _maybe_build_current_mesh() -> None:
            mode = gui_render.value
            if mode.startswith("Mesh") and st["codes"] is not None and not st["busy"]:
                _build_mesh(mode == "Mesh (low)")

        def _run_generate() -> None:
            if st["busy"]:
                return
            prompt = (gui_prompt.value or "").strip()
            if not prompt:
                gui_status.value = "enter a prompt first"
                return
            if not _GEN_LOCK.acquire(blocking=False):
                gui_status.value = "another client is generating; try again in a moment"
                return
            st["busy"] = True
            st["generating"] = True
            st["playing"] = True
            st["joints"] = None
            st["T"] = 1
            st["codes"] = None
            st["mesh"] = {}
            st["anchor"] = engine.canonical_anchor()
            gui_frame.value = 0
            t0 = time.time()
            ttft = None
            n_tok = 0
            try:
                for joints, _committed, n_tok, codes in engine.stream_generate(
                    prompt, cfg_scale=float(gui_cfg.value), temperature=float(gui_temp.value),
                    max_tok=int(gui_maxtok.value), chunk=int(gui_chunk.value), anchor=st["anchor"],
                ):
                    if ttft is None:
                        ttft = time.time() - t0
                    _set_buffer(joints)
                    st["codes"] = codes
                    gui_status.value = (f"generating… {n_tok} tok / {joints.shape[0]}f "
                                        f"({joints.shape[0] / engine.fps:.1f}s)  TTFT {ttft:.2f}s")
            except Exception as exc:  # noqa: BLE001
                gui_status.value = f"generation failed: {exc}"
                return
            finally:
                st["generating"] = False
                st["busy"] = False
                _GEN_LOCK.release()
            st["playing"] = True
            dur = 0 if st["joints"] is None else st["joints"].shape[0] / engine.fps
            gui_status.value = (f"done: {n_tok} tok / {st['T']}f ({dur:.1f}s), "
                                f"TTFT {ttft:.2f}s (chunk {int(gui_chunk.value)})")
            # If a mesh mode is selected, build it now (skeleton streamed during gen).
            _maybe_build_current_mesh()

        @gui_generate.on_click
        def _(_=None) -> None:
            threading.Thread(target=_run_generate, daemon=True).start()

        @gui_render.on_update
        def _(_=None) -> None:
            # Switching to a mesh mode after generation builds that LOD (once).
            threading.Thread(target=_maybe_build_current_mesh, daemon=True).start()
            _render(int(gui_frame.value))

        @gui_frame.on_update
        def _(_=None) -> None:
            _render(int(gui_frame.value))

        @gui_play.on_click
        def _(_=None) -> None:
            st["playing"] = not st["playing"]

        # Mesh vertex buffers are large; play them at ~25 fps (frame stride) so the
        # websocket keeps up (real-time is preserved by stepping `stride` frames).
        mesh_stride = max(1, round(engine.fps / 25.0))

        def play_loop() -> None:
            # Runs until a scene/gui op throws (client disconnected) -> break.
            while True:
                if st["T"] > 1 and (st["playing"] or st["generating"]) and st["joints"] is not None:
                    is_mesh = (gui_render.value != "Skeleton" and not st["generating"]
                               and st["mesh"].get(gui_render.value == "Mesh (low)") is not None)
                    stride = mesh_stride if is_mesh else 1
                    cap = st["T"]
                    nxt = int(gui_frame.value) + stride
                    if nxt >= cap:
                        nxt = cap - 1 if st["generating"] else 0
                    try:
                        gui_frame.value = nxt
                        _render(nxt)
                    except Exception:  # noqa: BLE001
                        break
                    time.sleep(stride / max(1e-3, engine.fps * gui_speed.value))
                else:
                    time.sleep(0.05)

        threading.Thread(target=play_loop, daemon=True).start()

    print(f"[live] hosting {args.checkpoint} (per-client sessions); open the URL above", flush=True)
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
