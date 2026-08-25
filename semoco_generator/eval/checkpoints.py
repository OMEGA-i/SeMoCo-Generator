"""Semoco checkpoint registry — the single source of truth for "which checkpoints are active."

A checkpoint is identified by its ``runs/`` directory name. All other paths
(codes_root, tokenizer, text encoder) are auto-discovered from the training
artifacts that already exist alongside the checkpoint.

.. code-block:: bash

    # Scan all eval-ready checkpoints → YAML snippets
    python -m semoco_generator.eval.checkpoints scan

    # List registered checkpoints
    python -m semoco_generator.eval.checkpoints list

The registry file at ``configs/eval_checkpoints.yaml`` is the user-editable
source of truth. Adding a checkpoint is adding its ``runs/`` dir name to the
file; removal is deleting the entry. No auto-discovery surprises.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..local_uri import resolve_local_uri

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckpointSpec:
    """Fully resolved paths and config for one Semoco checkpoint.

    Every field needed to construct an :class:`SemocoModel` is present. Callers
    never need to know where codes_root or text_encoder came from — they just
    use the fields.
    """

    name: str
    """Human-readable identifier (the ``runs/`` directory name)."""

    model_ckpt: Path
    """Path to the Semoco ``best.pt`` (or ``latest.pt``) checkpoint file."""

    tokenizer_ckpt: Path
    """Path to the frozen tokenizer checkpoint."""

    codes_root: Path
    """Path to the T2M code store root directory."""

    text_encoder: str
    """Text encoder key: ``"flan"``, ``"siglip"``, or ``"qwen3"``."""

    max_tok: int = 125
    """Default maximum motion tokens for generation."""


@dataclass
class Registry:
    """In-memory representation of ``configs/eval_checkpoints.yaml``."""

    checkpoints: dict[str, CheckpointSpec] = field(default_factory=dict)
    groups: dict[str, list[str]] = field(default_factory=dict)
    default: list[str] = field(default_factory=list)

    @property
    def names(self) -> list[str]:
        return sorted(self.checkpoints.keys())


# ---------------------------------------------------------------------------
# Discovery (hidden implementation)
# ---------------------------------------------------------------------------

# Directory name prefixes that identify training run directories.
_RUN_PREFIXES = ("t2m_", "mgpt_", "motion_gpt_")


def _find_best_ckpt(run_dir: Path) -> Path | None:
    """Return ``model/best.pt``, or ``model/latest.pt``, or None."""
    for name in ("best.pt", "latest.pt"):
        p = run_dir / "model" / name
        if p.is_file():
            return p
    return None


def discover_from_dir(run_dir: Path) -> CheckpointSpec | None:
    """Auto-discover a checkpoint spec from a ``runs/<name>/`` directory.

    Discovery chain::

        runs/<name>/config_resolved.json  →  codes_root, text_encoder
        runs/<name>/model/best.pt         →  model checkpoint
        {codes_root}/test.meta.json       →  tokenizer_checkpoint

    Returns ``None`` if any required artifact is missing.
    """
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        return None

    name = run_dir.name

    # Step 1: find model checkpoint
    model_ckpt = _find_best_ckpt(run_dir)
    if model_ckpt is None:
        return None

    # Step 2: read training config
    cfg_file = run_dir / "config_resolved.json"
    if not cfg_file.is_file():
        return None
    try:
        cfg = json.loads(cfg_file.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    # config.data.codes_root   (e.g. "local://t2m_codes")
    # config.text.encode       (e.g. "flan")
    config_section = cfg.get("config", {})
    data_section = config_section.get("data", {})
    text_section = config_section.get("text", {})

    codes_uri = data_section.get("codes_root")
    if codes_uri is None:
        return None
    codes_root = resolve_local_uri(codes_uri)

    text_encoder = text_section.get("encode")
    if text_encoder is None:
        # Fallback: try model.meta.encode_key from the checkpoint itself
        text_encoder = _read_encode_key_from_ckpt(model_ckpt)
    if text_encoder is None:
        return None

    # Step 3: discover tokenizer from code store meta.json
    # File is at {codes_root}/{split}.meta.json — try "test" first, then any split
    tokenizer_ckpt = _discover_tokenizer_ckpt(codes_root)
    if tokenizer_ckpt is None:
        return None

    return CheckpointSpec(
        name=name,
        model_ckpt=model_ckpt,
        tokenizer_ckpt=tokenizer_ckpt,
        codes_root=codes_root,
        text_encoder=text_encoder,
    )


def _read_encode_key_from_ckpt(ckpt_path: Path) -> str | None:
    """Read ``data_meta.encode_key`` or ``text_encoder_key`` from a checkpoint file."""
    try:
        import torch

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    meta = ckpt.get("data_meta") or {}
    key = meta.get("encode_key") or ckpt.get("text_encoder_key")
    if isinstance(key, str) and key:
        return key
    return None


def _discover_tokenizer_ckpt(codes_root: Path) -> Path | None:
    """Read ``tokenizer_checkpoint`` field from ``{split}.meta.json`` files."""
    for split in ("test", "val", "train"):
        meta_file = codes_root / f"{split}.meta.json"
        if not meta_file.is_file():
            continue
        try:
            meta = json.loads(meta_file.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        tok = meta.get("tokenizer_checkpoint")
        if tok:
            p = Path(tok)
            if p.is_file():
                return p
            # Path might be absolute and valid, or relative to something else.
            # If the stored path doesn't exist literally, try resolving against
            # the codes root's parent.
            if not p.is_absolute():
                p = (codes_root / tok).resolve()
                if p.is_file():
                    return p
            # Keep the original — caller will get a clear FileNotFoundError later.
            return p
    return None


# ---------------------------------------------------------------------------
# Scan helper (CLI and user-facing)
# ---------------------------------------------------------------------------


def scan_runs(runs_root: Path | None = None) -> list[dict[str, Any]]:
    """Scan ``runs/`` for all eval-ready checkpoints.

    Returns a list of dicts suitable for YAML serialization. Each dict has
    ``name``, ``checkpoint``, ``codes_root``, ``text_encoder``, and
    ``tokenizer`` fields — the user copy-pastes entries into the registry.
    """
    if runs_root is None:
        runs_root = _default_runs_root()
    else:
        runs_root = Path(runs_root)

    results: list[dict[str, Any]] = []
    if not runs_root.is_dir():
        return results

    for entry in sorted(runs_root.iterdir()):
        if not entry.is_dir():
            continue
        if not entry.name.startswith(_RUN_PREFIXES):
            continue
        spec = discover_from_dir(entry)
        if spec is None:
            continue
        results.append(
            {
                "name": spec.name,
                "checkpoint": str(_repo_relative(spec.model_ckpt)),
                "codes_root": str(spec.codes_root),
                "text_encoder": spec.text_encoder,
                "tokenizer": str(spec.tokenizer_ckpt),
            }
        )

    return results


def _default_runs_root() -> Path:
    """Return the repo ``runs/`` directory."""
    return Path(__file__).resolve().parents[2] / "runs"


def _repo_relative(p: Path) -> Path:
    """Try to make *p* relative to the repo root for readability."""
    try:
        repo = Path(__file__).resolve().parents[2]
        return p.resolve().relative_to(repo)
    except ValueError:
        return p


# ---------------------------------------------------------------------------
# Registry loading
# ---------------------------------------------------------------------------


def _default_registry_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "eval_checkpoints.yaml"


def load_registry(path: str | Path | None = None) -> Registry:
    """Load the checkpoint registry from ``configs/eval_checkpoints.yaml``.

    Returns an empty :class:`Registry` if the file doesn't exist.
    """
    if path is None:
        path = _default_registry_path()
    else:
        path = Path(path)

    if not path.is_file():
        return Registry()

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    registry = Registry()
    raw_checkpoints = raw.get("checkpoints") or {}
    for name, entry in raw_checkpoints.items():
        spec = _parse_entry(name, entry)
        if spec is not None:
            registry.checkpoints[name] = spec

    raw_groups = raw.get("groups") or {}
    for group_name, group_names in raw_groups.items():
        if isinstance(group_names, list):
            registry.groups[group_name] = [str(n) for n in group_names]

    raw_default = raw.get("default") or []
    if isinstance(raw_default, list):
        registry.default = [str(n) for n in raw_default]

    return registry


def _parse_entry(name: str, entry: Any) -> CheckpointSpec | None:
    """Parse a single checkpoint entry from the YAML registry.

    Two forms are supported:

    1. **Auto-discover** (just a checkpoint path)::

        my_checkpoint:
          checkpoint: runs/my_checkpoint/model/best.pt

    2. **Explicit** (all fields specified)::

        t2m_custom:
          checkpoint: /path/to/model.pt
          codes_root: local://t2m_codes
          tokenizer: /path/to/tokenizer.pt
          text_encoder: flan
    """
    if isinstance(entry, str):
        # Short form: just a path string
        ckpt_path = _resolve_repo_path(Path(entry))
        run_dir = ckpt_path.parent.parent  # model/best.pt → runs/<name>
        return discover_from_dir(run_dir)

    if not isinstance(entry, dict):
        return None

    ckpt_path = _resolve_repo_path(Path(entry.get("checkpoint", "")))
    model_ckpt = ckpt_path if ckpt_path.is_file() else None

    if model_ckpt is None:
        return None

    # If codes_root / text_encoder / tokenizer are explicitly provided, use them.
    # Otherwise, try auto-discovery from the run directory.
    explicit_codes = entry.get("codes_root")
    explicit_encoder = entry.get("text_encoder")
    explicit_tokenizer = entry.get("tokenizer")

    if explicit_codes and explicit_encoder and explicit_tokenizer:
        # Fully explicit — no discovery needed
        codes_root = resolve_local_uri(str(explicit_codes))
        return CheckpointSpec(
            name=name,
            model_ckpt=model_ckpt,
            tokenizer_ckpt=_resolve_repo_path(Path(explicit_tokenizer)),
            codes_root=codes_root,
            text_encoder=str(explicit_encoder),
        )

    # Partial or no explicit fields — try auto-discovery from the run dir
    run_dir = model_ckpt.parent.parent  # model/best.pt → runs/<name>
    discovered = discover_from_dir(run_dir)
    if discovered is None:
        return None

    # Override discovered values with any explicit fields
    return CheckpointSpec(
        name=name,
        model_ckpt=model_ckpt,
        tokenizer_ckpt=(
            _resolve_repo_path(Path(explicit_tokenizer))
            if explicit_tokenizer
            else discovered.tokenizer_ckpt
        ),
        codes_root=(
            resolve_local_uri(str(explicit_codes))
            if explicit_codes
            else discovered.codes_root
        ),
        text_encoder=(
            str(explicit_encoder) if explicit_encoder else discovered.text_encoder
        ),
    )


def _resolve_repo_path(p: Path) -> Path:
    """Resolve a path that may be relative to the repo root."""
    if p.is_absolute():
        return p
    repo = Path(__file__).resolve().parents[2]
    resolved = (repo / p).resolve()
    if resolved.exists():
        return resolved
    return p


# ---------------------------------------------------------------------------
# SPEC resolution
# ---------------------------------------------------------------------------


def resolve(registry: Registry, spec: str) -> list[CheckpointSpec]:
    """Parse a SPEC string against *registry*.

    SPEC forms:

    +---------------------------+-------------------------------------------+
    | SPEC                      | Meaning                                   |
    +===========================+===========================================+
    | ``name``                  | Single checkpoint by registry name        |
    +---------------------------+-------------------------------------------+
    | ``@group``                | All checkpoints in a named group          |
    +---------------------------+-------------------------------------------+
    | ``a,b,c``                 | Comma-separated names / groups / ``all``  |
    +---------------------------+-------------------------------------------+
    | ``all``                   | All checkpoints in the registry           |
    +---------------------------+-------------------------------------------+
    | ``runs/.../best.pt``      | Raw path (one-off, no registry needed)    |
    +---------------------------+-------------------------------------------+
    """
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if not parts:
        return []

    # If the only part is "all", return everything sorted.
    if len(parts) == 1 and parts[0] == "all":
        return sorted(registry.checkpoints.values(), key=lambda s: s.name)

    results: list[CheckpointSpec] = []
    seen: set[str] = set()

    for part in parts:
        if part == "all":
            for spec_obj in sorted(registry.checkpoints.values(), key=lambda s: s.name):
                if spec_obj.name not in seen:
                    results.append(spec_obj)
                    seen.add(spec_obj.name)
        elif part.startswith("@"):
            # Group reference
            group_name = part[1:]
            group_names = registry.groups.get(group_name)
            if group_names is None:
                raise KeyError(
                    f"Unknown group '{group_name}'. "
                    f"Available groups: {sorted(registry.groups.keys())}"
                )
            for name in group_names:
                if name in seen:
                    continue
                spec_obj = registry.checkpoints.get(name)
                if spec_obj is None:
                    raise KeyError(
                        f"Checkpoint '{name}' (in group '{group_name}') "
                        f"not found in registry."
                    )
                results.append(spec_obj)
                seen.add(name)
        elif part.startswith("runs/") or part.startswith("/"):
            # Raw path
            ckpt_path = Path(part)
            if not ckpt_path.is_absolute():
                ckpt_path = _resolve_repo_path(ckpt_path)
            run_dir = ckpt_path.parent.parent  # model/best.pt → runs/<name>
            spec_obj = discover_from_dir(run_dir)
            if spec_obj is None:
                raise FileNotFoundError(
                    f"Cannot auto-discover checkpoint from '{part}' — "
                    f"missing config_resolved.json or code store metadata."
                )
            results.append(spec_obj)
        else:
            # Registry name
            spec_obj = registry.checkpoints.get(part)
            if spec_obj is None:
                raise KeyError(
                    f"Unknown checkpoint '{part}'. "
                    f"Available: {sorted(registry.checkpoints.keys())}"
                )
            if part not in seen:
                results.append(spec_obj)
                seen.add(part)

    return results


def resolve_default(registry: Registry) -> list[CheckpointSpec]:
    """Resolve the registry's ``default`` list to specs."""
    if not registry.default:
        return []
    specs: list[CheckpointSpec] = []
    for name in registry.default:
        s = registry.checkpoints.get(name)
        if s is not None:
            specs.append(s)
    return specs


def configured_default_specs(path: str | Path | None = None) -> list[CheckpointSpec]:
    """Return the fully resolved checkpoint specs selected by the registry.

    This is the single interface for callers that need the local assets for a
    normal Semoco evaluation without choosing a particular model themselves.
    It deliberately returns only the registry's explicit ``default`` entries;
    an empty result means the deployment has not configured a default run.
    """
    return resolve_default(load_registry(path))


# ---------------------------------------------------------------------------
# CLI (``python -m semoco_generator.eval.checkpoints scan|list``)
# ---------------------------------------------------------------------------


def _cli_scan() -> None:
    """Print YAML-ready registry entries for all discoverable checkpoints."""
    entries = scan_runs()
    if not entries:
        print("# No eval-ready checkpoints found in runs/")
        return

    print("# Auto-generated by: python -m semoco_generator.eval.checkpoints scan")
    print("# Copy entries you want into configs/eval_checkpoints.yaml")
    print()
    print("checkpoints:")
    for e in entries:
        print(f"  {e['name']}:")
        print(f"    checkpoint: {e['checkpoint']}")
        # Only print explicit fields when discovery might differ from defaults
    print()
    if entries:
        print("# groups:")
        print("#   quick:")
        print(f"#     - {entries[0]['name']}")


def _cli_list() -> None:
    """Print registered checkpoints from the registry file."""
    registry = load_registry()
    if not registry.checkpoints:
        path = _default_registry_path()
        # An entry whose .pt is missing is dropped during parsing, so an empty
        # registry on a fresh clone means "not trained yet", not "misconfigured".
        # Say which it is instead of leaving the user to guess.
        declared: list[str] = []
        if path.is_file():
            with open(path) as f:
                declared = list((yaml.safe_load(f) or {}).get("checkpoints") or {})
        if declared:
            print(
                f"No usable checkpoints. {path} declares {len(declared)} entry(ies) "
                f"({', '.join(declared)}) but none of their checkpoint files exist "
                f"yet — train a model, or point an entry at an existing .pt."
            )
        else:
            print(f"No checkpoints declared in {path}")
        return

    print(f"{len(registry.checkpoints)} checkpoint(s) registered:")
    for name in registry.names:
        spec = registry.checkpoints[name]
        print(f"  {name}")
        print(f"    codes:    {spec.codes_root.name}")
        print(f"    encoder:  {spec.text_encoder}")
        print(f"    tokenizer: {spec.tokenizer_ckpt}")

    if registry.groups:
        print(f"\n{len(registry.groups)} group(s):")
        for gname, gnames in sorted(registry.groups.items()):
            print(f"  @{gname}: {gnames}")

    if registry.default:
        print(f"\ndefault: {registry.default}")


def main() -> None:
    import sys

    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("usage: python -m semoco_generator.eval.checkpoints [scan|list]")
        print("  scan   Scan runs/ for eval-ready checkpoints (YAML snippets)")
        print("  list   List registered checkpoints from configs/eval_checkpoints.yaml")
        return

    cmd = sys.argv[1]
    if cmd == "scan":
        _cli_scan()
    elif cmd == "list":
        _cli_list()
    else:
        print(f"Unknown command: {cmd}")
        print("usage: python -m semoco_generator.eval.checkpoints [scan|list]")


if __name__ == "__main__":
    main()


__all__ = [
    "CheckpointSpec",
    "Registry",
    "discover_from_dir",
    "configured_default_specs",
    "load_registry",
    "resolve",
    "resolve_default",
    "scan_runs",
]
