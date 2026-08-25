"""TMR embedding and joint-quality metrics."""
from __future__ import annotations
import numpy as np
from ...metrics import compute_embedding_metrics, estimate_foot_contact, foot_skate, joint_jerk

# SOMA77 foot contact joints (from data.umr_schema.FOOT_CONTACT_SOMA77_INDICES).
SOMA77_FOOT_INDICES = (69, 70, 74, 75)


def score_embeddings(
    gen: np.ndarray,
    gt: np.ndarray,
    *,
    text: np.ndarray | None = None,
    retrieval_protocol: str = "full_gallery",
    seed: int = 0,
    kimodo_metrics: bool = False,
) -> dict[str, float]:
    return compute_embedding_metrics(
        gen_emb=gen,
        gt_emb=gt,
        text_emb=text,
        retrieval_protocol=retrieval_protocol,
        seed=seed,
        kimodo_metrics=kimodo_metrics,
    )


def motion_quality(
    joints: np.ndarray,
    fps: float,
    foot_indices=SOMA77_FOOT_INDICES,
) -> dict[str, float]:
    """Local engineering foot-skate / jerk on SOMA77 joints.

    Not equivalent to Kimodo official contact metrics unless ``foot_contacts``
    from ``motion.npz`` are available.
    """
    j = np.asarray(joints)
    if j.ndim != 3:
        raise ValueError(f"expected joints [T,J,3], got {j.shape}")
    idx = list(foot_indices)
    if j.shape[1] < max(idx) + 1:
        # Fall back silently for non-SOMA skeletons rather than indexing wrong joints.
        return {"foot_skate": float("nan"), "jerk": joint_jerk(j, fps)}
    contact = estimate_foot_contact(j, idx, fps)
    return {
        "foot_skate": foot_skate(j, contact, idx, fps),
        "jerk": joint_jerk(j, fps),
    }


__all__ = ["SOMA77_FOOT_INDICES", "motion_quality", "score_embeddings"]
