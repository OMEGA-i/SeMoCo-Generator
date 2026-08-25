"""Score motion predictions: future-window ADE / FDE.

Reads ``{cid}_pred.npy`` from ``predict`` and the matching ``{cid}_gt.npy``
from ``build_gt``, and reports the test-set mean of both metrics in metres.

For best-of-K runs, score ``_pred.npy`` for min-ADE and re-run with
``--pred-suffix _predfde.npy`` for min-FDE, since the two picks are different
samples.

Example::

    python -m semoco_generator.eval.prediction.score \\
        --pred-dir local://pred_out/test --gt-dir local://pred_gt/test \\
        --out-json runs/prediction/metrics.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ...local_uri import resolve_local_uri
from .metrics import DEFAULT_PREDICT_RATIO, ade, fde


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--gt-dir", required=True, help="dir of {cid}_gt.npy from build_gt")
    ap.add_argument("--pred-suffix", default="_pred.npy",
                    help="use _predfde.npy to score the min-FDE pick of a best-of-K run")
    ap.add_argument("--predict-ratio", type=float, default=DEFAULT_PREDICT_RATIO,
                    help="must match the value used to generate the predictions")
    ap.add_argument("--rec-id-list", default=None,
                    help="restrict to these ids (cids derived via / -> __, space -> _)")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--tag", default="prediction")
    args = ap.parse_args()

    pred_dir = resolve_local_uri(args.pred_dir)
    gt_dir = resolve_local_uri(args.gt_dir)
    restrict = None
    if args.rec_id_list:
        restrict = {ln.strip().replace("/", "__").replace(" ", "_")
                    for ln in Path(args.rec_id_list).read_text().splitlines() if ln.strip()}

    ades: list[float] = []
    fdes: list[float] = []
    n_missing_gt = 0
    n_bad = 0
    t0 = time.time()
    for pf in pred_dir.glob(f"*{args.pred_suffix}"):
        cid = pf.name[: -len(args.pred_suffix)]
        if restrict is not None and cid not in restrict:
            continue
        gf = gt_dir / f"{cid}_gt.npy"
        if not gf.exists():
            n_missing_gt += 1
            continue
        try:
            pred = np.load(pf)
            gt = np.load(gf)
        except (EOFError, ValueError, OSError):
            n_bad += 1
            continue
        a = ade(pred, gt, args.predict_ratio)
        d = fde(pred, gt)
        if np.isfinite(a):
            ades.append(a)
        if np.isfinite(d):
            fdes.append(d)

    out = {
        "tag": args.tag,
        "pred_dir": str(pred_dir),
        "gt_dir": str(gt_dir),
        "pred_suffix": args.pred_suffix,
        "predict_ratio": args.predict_ratio,
        "n": len(ades),
        "n_missing_gt": n_missing_gt,
        "n_bad": n_bad,
        "ADE": float(np.mean(ades)) if ades else None,
        "FDE": float(np.mean(fdes)) if fdes else None,
        "ADE_std": float(np.std(ades)) if ades else None,
        "FDE_std": float(np.std(fdes)) if fdes else None,
    }
    out_json = resolve_local_uri(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2))
    if not ades:
        raise SystemExit(f"[pred-score] no scorable clips (missing_gt={n_missing_gt}, bad={n_bad})")
    print(f"[pred-score] {args.tag}: n={out['n']} ADE={out['ADE']:.4f} FDE={out['FDE']:.4f} "
          f"(missing_gt={n_missing_gt}, {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
