"""Regression tests for the patches we apply on top of the kimodo submodule."""

import pytest
import torch

pytestmark = pytest.mark.requires_kimodo


class _Rep:
    """Stand-in carrying only what ``translate_2d`` reads off ``self``."""

    slice_dict = {"root_pos": slice(0, 3)}


def _translate(batch: int, frames: int):
    from kimodo.motion_rep.reps.tmr_motionrep import TMRMotionRep

    from semoco_generator.eval.tmr.kimodo_compat import patch_tmr_motion_rep

    patch_tmr_motion_rep()
    features = torch.zeros(batch, frames, 3)
    offsets = torch.arange(batch * 2, dtype=torch.float32).reshape(batch, 2)
    return TMRMotionRep.translate_2d(_Rep(), features, offsets), offsets


# batch == frames is the case upstream gets wrong without raising.
@pytest.mark.parametrize("batch, frames", [(1, 10), (4, 10), (10, 10), (64, 300)])
def test_translate_2d_offsets_each_clip_independently(batch, frames):
    out, offsets = _translate(batch, frames)

    for b in range(batch):
        assert torch.allclose(out[b, :, 0], offsets[b, 0].expand(frames))
        assert torch.allclose(out[b, :, 2], offsets[b, 1].expand(frames))
    assert torch.allclose(out[:, :, 1], torch.zeros(batch, frames)), "y axis must not move"


def test_patch_is_idempotent():
    from semoco_generator.eval.tmr.kimodo_compat import patch_tmr_motion_rep

    patch_tmr_motion_rep()
    patch_tmr_motion_rep()

    out, offsets = _translate(4, 10)
    assert torch.allclose(out[0, :, 0], offsets[0, 0].expand(10))
