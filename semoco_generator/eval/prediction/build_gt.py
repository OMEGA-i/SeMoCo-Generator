"""Build ground-truth joints for motion prediction: FK of the real UMR-499 features.

For each requested ``rec_id`` this reads the clip's real ``features [T, 499]``
plus its frame-0 anchor from the release parquet and runs the SOMA-X forward
kinematics directly (``materialize_features_matrices`` ->
``soma77_joints_world_xyz``), then takes the SMPL-22 joint subset.

This ground truth carries no tokenizer reconstruction floor: it never passes
through the codebook, so ADE/FDE against it measure generation error against the
motion the model was trained to reproduce. Decoding the ground-truth *codes*
instead would share the quantization floor with the prediction and under-report
the error.

Streams the parquet at constant memory, writing each clip as it is found, and
stops early once every requested id exists. Runs on CPU.

Example::

    python -m semoco_generator.eval.prediction.build_gt \\
        --parquet-dir <release>/derived_umr_<hash> --split test \\
        --out-dir local://pred_gt/test
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from ...dataset.umr_parquet import iter_parquet_rows
from ...local_uri import resolve_local_uri
from ...paths import ensure_tokenizer_on_path

_COLS = ["features", "init_root_pos", "init_root_rot6d", "init_joints76_rot6d", "identity_coeffs"]


def safe_cid(rec_id: str) -> str:
    """Filesystem-safe clip id shared by every file this pipeline writes."""
    return rec_id.replace("/", "__").replace(" ", "_")


def _fk_smpl22(row, soma77_to_smpl22, CanonicalAnchor,
               materialize_features_matrices, soma77_joints_world_xyz_from_matrices):
    feats = np.asarray(row["features"], dtype=np.float32)
    anchor = CanonicalAnchor(
        init_root_pos=np.asarray(row["init_root_pos"], dtype=np.float32).reshape(3),
        init_root_rot6d=np.asarray(row["init_root_rot6d"], dtype=np.float32).reshape(6),
        init_joints76_rot6d=np.asarray(row["init_joints76_rot6d"], dtype=np.float32).reshape(76, 6),
    )
    identity = np.asarray(row["identity_coeffs"], dtype=np.float32)
    if identity.ndim == 1:
        identity = identity.reshape(1, -1)
    mats = materialize_features_matrices(feats, anchor)
    joints77 = soma77_joints_world_xyz_from_matrices(
        mats.rotmat77, mats.transl, identity, device="cpu",
    )
    joints77 = np.asarray(joints77[1:], dtype=np.float32)  # drop the frame-0 seed, as the pred path does
    return soma77_to_smpl22(joints77).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet-dir", required=True, help="<release>/derived_umr_<hash>")
    ap.add_argument("--split", default="test")
    ap.add_argument("--rec-id-list", default=None,
                    help="one rec_id per line; default = every clip in the split")
    ap.add_argument("--from-pred-dir", default=None,
                    help="derive the wanted ids from *_pred.npy filenames in this dir, "
                         "for when a rec_id list is not handy")
    ap.add_argument("--out-dir", required=True, help="dir for {cid}_gt.npy (accepts local:// URIs)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    ensure_tokenizer_on_path()
    from data.soma77_fk import soma77_joints_world_xyz_from_matrices  # noqa: WPS433
    from data.umr_schema import CanonicalAnchor  # noqa: WPS433
    from data.umr_to_soma77 import materialize_features_matrices  # noqa: WPS433
    from eval.recon_common import soma77_to_smpl22  # type: ignore  # noqa: WPS433

    out_dir = resolve_local_uri(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    want = None
    want_cids: set[str] = set()
    if args.rec_id_list:
        want = {ln.strip() for ln in Path(args.rec_id_list).read_text().splitlines() if ln.strip()}
    elif args.from_pred_dir:
        pred_dir = resolve_local_uri(args.from_pred_dir)
        want_cids = {f.name[: -len("_pred.npy")] for f in pred_dir.glob("*_pred.npy")}
        print(f"[pred-gt] {len(want_cids)} cids from pred dir", flush=True)

    already = set() if args.overwrite else {
        f.name[: -len("_gt.npy")] for f in out_dir.glob("*_gt.npy")
    }

    n_written, n_seen, t0 = 0, 0, time.time()
    target = len(want) if want is not None else (len(want_cids) or None)
    for row in iter_parquet_rows(args.parquet_dir, args.split, cols=_COLS):
        rid = row["rec_id"]
        cid = safe_cid(rid)
        if want is not None:
            if rid not in want:
                continue
        elif want_cids and cid not in want_cids:
            continue
        n_seen += 1
        if cid in already:
            if target and n_seen >= target:
                break
            continue
        gt22 = _fk_smpl22(row, soma77_to_smpl22, CanonicalAnchor,
                          materialize_features_matrices, soma77_joints_world_xyz_from_matrices)
        np.save(out_dir / f"{cid}_gt.npy", gt22)
        n_written += 1
        if n_written % 2000 == 0:
            print(f"[pred-gt] wrote {n_written} ({n_seen} seen)  {time.time() - t0:.0f}s", flush=True)
        if target and n_seen >= target:
            break
    print(f"[pred-gt] done: wrote {n_written}, seen {n_seen} -> {out_dir} "
          f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
