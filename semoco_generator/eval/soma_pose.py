"""Validated full-pose data carried by normal SOMA77 motion clips."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schema import MotionClip


SEMOCO_POSE_CONVENTION = "relative_to_soma_joint_orient"
KIMODO_POSE_CONVENTION = "kimodo_somaskel77_local_rotation_matrix"

SEMOCO_ROTMAT_AUX_KEY = "soma_rotmat77"
KIMODO_ROTMAT_AUX_KEY = "soma_local_rotmat77"
ROOT_TRANSLATION_AUX_KEY = "soma_root_translation"
IDENTITY_AUX_KEY = "soma_identity_coeffs"
FOOT_CONTACTS_AUX_KEY = "soma_foot_contacts"

SOMA_FOOT_INDICES = (69, 70, 74, 75)
SOMA_TOE_BASE_INDICES = (70, 75)
SOMA_TOE_END_INDICES = (71, 76)
MAX_FOOT_SPEED_MPS = 10.0
MAX_TOE_ANGULAR_VELOCITY_DEG_S = 900.0
MIN_TOE_SEGMENT_LENGTH_M = 1e-4


@dataclass(frozen=True)
class SomaPose:
    """One complete SOMA77 pose sequence from a model's normal output path.

    The rotation arrays have the same shape for Semoco and KiMoDo, but their
    local-frame conventions differ.  ``pose_convention`` is therefore part of
    the interface and callers must convert through the appropriate official
    rig semantics before comparing or transferring rotations.
    """

    rotmat77: np.ndarray
    transl: np.ndarray
    joints77: np.ndarray
    fps: float
    pose_convention: str
    identity_coeffs: np.ndarray | None = None
    foot_contacts: np.ndarray | None = None

    def __post_init__(self) -> None:
        rotations = np.asarray(self.rotmat77, dtype=np.float32)
        translation = np.asarray(self.transl, dtype=np.float32)
        joints = np.asarray(self.joints77, dtype=np.float32)
        frames = rotations.shape[0] if rotations.ndim else 0

        if rotations.ndim != 4 or rotations.shape[1:] != (77, 3, 3):
            raise ValueError(f"rotmat77 must be [T,77,3,3], got {rotations.shape}")
        if frames < 1:
            raise ValueError("rotmat77 must contain at least one frame")
        if translation.shape != (frames, 3):
            raise ValueError(f"transl must be [{frames},3], got {translation.shape}")
        if joints.shape != (frames, 77, 3):
            raise ValueError(f"joints77 must be [{frames},77,3], got {joints.shape}")
        if not all(np.isfinite(value).all() for value in (rotations, translation, joints)):
            raise ValueError("SOMA pose contains non-finite values")
        if not np.isfinite(float(self.fps)) or float(self.fps) <= 0.0:
            raise ValueError(f"fps must be finite and positive, got {self.fps!r}")
        if self.pose_convention not in {SEMOCO_POSE_CONVENTION, KIMODO_POSE_CONVENTION}:
            raise ValueError(f"unsupported SOMA pose convention {self.pose_convention!r}")

        gram = np.swapaxes(rotations, -1, -2) @ rotations
        eye = np.eye(3, dtype=np.float32)
        orth_error = float(np.max(np.abs(gram - eye)))
        det_error = float(np.max(np.abs(np.linalg.det(rotations) - 1.0)))
        if orth_error > 3e-3 or det_error > 3e-3:
            raise ValueError(
                "rotmat77 must contain proper rotation matrices "
                f"(orth_error={orth_error:.3g}, det_error={det_error:.3g})"
            )

        identity = self.identity_coeffs
        if identity is not None:
            identity = np.asarray(identity, dtype=np.float32)
            if identity.ndim == 1:
                identity = identity.reshape(1, -1)
            if identity.ndim != 2 or identity.shape[0] != 1 or not np.isfinite(identity).all():
                raise ValueError(f"identity_coeffs must be finite [C] or [1,C], got {identity.shape}")

        contacts = self.foot_contacts
        if contacts is not None:
            contacts = np.asarray(contacts, dtype=np.float32)
            if contacts.shape != (frames, 4) or not np.isfinite(contacts).all():
                raise ValueError(f"foot_contacts must be finite [{frames},4], got {contacts.shape}")

        object.__setattr__(self, "rotmat77", rotations)
        object.__setattr__(self, "transl", translation)
        object.__setattr__(self, "joints77", joints)
        object.__setattr__(self, "identity_coeffs", identity)
        object.__setattr__(self, "foot_contacts", contacts)

    @property
    def num_frames(self) -> int:
        return int(self.rotmat77.shape[0])

    @classmethod
    def from_motion_clip(cls, clip: MotionClip, *, model_id: str) -> "SomaPose":
        """Read a full pose from a normal converted ``soma77`` clip."""
        if clip.rep != "soma77":
            raise ValueError(f"full SOMA pose requires soma77, got {clip.rep!r}")
        if model_id == "semoco":
            rotation_key = SEMOCO_ROTMAT_AUX_KEY
            convention = SEMOCO_POSE_CONVENTION
        elif model_id == "kimodo":
            rotation_key = KIMODO_ROTMAT_AUX_KEY
            convention = KIMODO_POSE_CONVENTION
        else:
            raise ValueError(f"model {model_id!r} does not expose a full SOMA pose")
        missing = [
            key for key in (rotation_key, ROOT_TRANSLATION_AUX_KEY)
            if key not in clip.aux
        ]
        if missing:
            raise ValueError(
                f"{model_id} converted clip is missing full-pose aux: "
                + ", ".join(missing)
            )
        return cls(
            rotmat77=clip.aux[rotation_key],
            transl=clip.aux[ROOT_TRANSLATION_AUX_KEY],
            joints77=clip.array,
            fps=float(clip.fps),
            pose_convention=convention,
            identity_coeffs=clip.aux.get(IDENTITY_AUX_KEY),
            foot_contacts=clip.aux.get(FOOT_CONTACTS_AUX_KEY),
        )


def semoco_pose_aux(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Map tokenizer full-pose output onto the durable ``MotionClip.aux`` keys."""
    return {
        SEMOCO_ROTMAT_AUX_KEY: np.asarray(arrays["rotmat77"], dtype=np.float32),
        ROOT_TRANSLATION_AUX_KEY: np.asarray(arrays["transl"], dtype=np.float32),
        IDENTITY_AUX_KEY: np.asarray(arrays["identity_coeffs"], dtype=np.float32),
        FOOT_CONTACTS_AUX_KEY: np.asarray(arrays["foot_contacts"], dtype=np.float32),
    }


def soma_pose_dynamics(joints77: np.ndarray, fps: float) -> dict[str, float]:
    """Measure source-rate foot continuity before resampling or fitting."""
    joints = np.asarray(joints77, dtype=np.float32)
    if joints.ndim != 3 or joints.shape[1:] != (77, 3) or joints.shape[0] < 1:
        raise ValueError(f"expected non-empty joints77 [T,77,3], got {joints.shape}")
    if not np.isfinite(joints).all():
        raise ValueError("joints77 contains non-finite values")
    if not np.isfinite(float(fps)) or float(fps) <= 0.0:
        raise ValueError(f"invalid source fps {fps!r}")

    toes = joints[:, SOMA_TOE_END_INDICES] - joints[:, SOMA_TOE_BASE_INDICES]
    toe_lengths = np.linalg.norm(toes, axis=-1)
    if joints.shape[0] < 2:
        return {
            "source_fps": float(fps),
            "raw_max_foot_step_mm": 0.0,
            "raw_max_foot_speed_mps": 0.0,
            "raw_max_toe_turn_deg": 0.0,
            "raw_max_toe_angular_velocity_deg_s": 0.0,
            "raw_min_toe_segment_mm": float(toe_lengths.min() * 1000.0),
        }

    foot_steps = np.linalg.norm(np.diff(joints[:, SOMA_FOOT_INDICES], axis=0), axis=-1)
    unit_toes = toes / np.maximum(toe_lengths[..., None], 1e-8)
    cosine = np.sum(unit_toes[1:] * unit_toes[:-1], axis=-1).clip(-1.0, 1.0)
    toe_turn_deg = np.rad2deg(np.arccos(cosine))
    max_step_m = float(foot_steps.max(initial=0.0))
    max_turn_deg = float(toe_turn_deg.max(initial=0.0))
    return {
        "source_fps": float(fps),
        "raw_max_foot_step_mm": max_step_m * 1000.0,
        "raw_max_foot_speed_mps": max_step_m * float(fps),
        "raw_max_toe_turn_deg": max_turn_deg,
        "raw_max_toe_angular_velocity_deg_s": max_turn_deg * float(fps),
        "raw_min_toe_segment_mm": float(toe_lengths.min() * 1000.0),
    }


def soma_pose_dynamics_failures(dynamics: dict[str, float]) -> list[str]:
    """Return stable quality-gate reasons for implausible foot motion."""
    values = [float(value) for key, value in dynamics.items() if key != "source_fps"]
    if not all(np.isfinite(value) for value in values):
        return ["non_finite:raw_foot_dynamics"]
    failures: list[str] = []
    if dynamics["raw_min_toe_segment_mm"] < MIN_TOE_SEGMENT_LENGTH_M * 1000.0:
        failures.append("quality_gate:raw_toe_segment_length")
    if dynamics["raw_max_foot_speed_mps"] > MAX_FOOT_SPEED_MPS:
        failures.append("quality_gate:raw_foot_speed_mps")
    if (
        dynamics["raw_max_toe_angular_velocity_deg_s"]
        > MAX_TOE_ANGULAR_VELOCITY_DEG_S
    ):
        failures.append("quality_gate:raw_toe_angular_velocity_deg_s")
    return failures


def audit_kimodo_pose_fk(
    pose: SomaPose,
    *,
    device: str = "cpu",
    max_error_m: float = 1e-4,
) -> dict[str, float | int | bool]:
    """Verify KiMoDo rotations against its official SOMASkeleton77 FK."""
    if pose.pose_convention != KIMODO_POSE_CONVENTION:
        raise ValueError("KiMoDo FK audit requires the KiMoDo pose convention")

    import torch

    from kimodo.skeleton import SOMASkeleton77

    skeleton = SOMASkeleton77().to(device).eval()
    with torch.inference_mode():
        local = torch.as_tensor(pose.rotmat77, dtype=torch.float32, device=device)
        root = torch.as_tensor(pose.transl, dtype=torch.float32, device=device)
        _global, recovered, _without_root = skeleton.fk(local, root)
    recovered_np = recovered.cpu().numpy().astype(np.float32, copy=False)
    delta = recovered_np - pose.joints77
    root_delta = recovered_np[:, skeleton.root_idx] - pose.transl
    max_error = float(np.abs(delta).max())
    return {
        "num_frames": pose.num_frames,
        "max_abs_error_m": max_error,
        "mean_abs_error_m": float(np.abs(delta).mean()),
        "rmse_m": float(np.sqrt(np.square(delta.astype(np.float64)).mean())),
        "max_root_error_m": float(np.abs(root_delta).max()),
        "all_finite": bool(np.isfinite(recovered_np).all()),
        "passed": bool(np.isfinite(recovered_np).all() and max_error <= float(max_error_m)),
    }


__all__ = [
    "FOOT_CONTACTS_AUX_KEY",
    "IDENTITY_AUX_KEY",
    "KIMODO_POSE_CONVENTION",
    "KIMODO_ROTMAT_AUX_KEY",
    "SEMOCO_POSE_CONVENTION",
    "SEMOCO_ROTMAT_AUX_KEY",
    "ROOT_TRANSLATION_AUX_KEY",
    "SomaPose",
    "audit_kimodo_pose_fk",
    "semoco_pose_aux",
    "soma_pose_dynamics",
    "soma_pose_dynamics_failures",
]
