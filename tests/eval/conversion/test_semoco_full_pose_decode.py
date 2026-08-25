"""Focused tests for full Semoco token decode without loading a real codec."""

from __future__ import annotations

import numpy as np

from semoco_generator.eval.conversions import ConversionContext, build_default_graph
from semoco_generator.eval.schema import MotionClip
from semoco_generator.eval.soma_pose import SomaPose
from semoco_generator.paths import ensure_tokenizer_on_path
from semoco_generator.tokenizer_bridge import FrozenMotionTokenizer

import pytest

pytestmark = pytest.mark.requires_tokenizer


def _identity6() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)


def _anchor() -> dict[str, np.ndarray]:
    return {
        "init_root_pos": np.array([0.0, 0.7, 0.0], dtype=np.float32),
        "init_root_rot6d": _identity6(),
        "init_joints76_rot6d": np.tile(_identity6(), (76, 1)),
    }


def _features(frames: int = 2) -> np.ndarray:
    ensure_tokenizer_on_path()
    from data.umr_schema import IDX_ROOT_Y, SLICE_JOINTS76_ROT6D, SLICE_ROOT_ROT6D, SLICE_ROOT_TRAJ

    values = np.zeros((frames, 499), dtype=np.float32)
    values[:, SLICE_ROOT_ROT6D] = _identity6()
    values[:, SLICE_JOINTS76_ROT6D] = np.tile(_identity6(), 76)
    values[:, SLICE_ROOT_TRAJ.start + IDX_ROOT_Y] = np.arange(frames, dtype=np.float32) + 1.0
    return values


def test_full_pose_decode_keeps_all_rotations_and_fk_joints(monkeypatch) -> None:
    ensure_tokenizer_on_path()
    from data import soma77_fk

    def fake_fk(rotmat77, transl, identity_coeffs, *, device):
        assert rotmat77.shape == (3, 77, 3, 3)  # anchor plus two decoded frames
        assert transl.shape == (3, 3)
        assert identity_coeffs.shape == (1, 10)
        joints = np.zeros((3, 77, 3), dtype=np.float32)
        joints[:, 0] = transl
        return joints

    monkeypatch.setattr(soma77_fk, "soma77_joints_world_xyz_from_matrices", fake_fk)
    bridge = object.__new__(FrozenMotionTokenizer)
    bridge.decode = lambda _codes: _features()  # type: ignore[method-assign]
    anchor = _anchor()

    result = bridge.decode_to_full_pose_arrays(
        np.zeros((1, 16), dtype=np.int64),
        init_root_pos=anchor["init_root_pos"],
        init_root_rot6d=anchor["init_root_rot6d"],
        init_joints76_rot6d=anchor["init_joints76_rot6d"],
        identity_coeffs=np.zeros(10, dtype=np.float32),
        device="cpu",
    )

    assert result["rotmat77"].shape == (2, 77, 3, 3)
    assert result["transl"].shape == (2, 3)
    assert result["joints77"].shape == (2, 77, 3)
    assert result["identity_coeffs"].shape == (1, 10)
    assert result["foot_contacts"].shape == (2, 4)
    np.testing.assert_allclose(
        result["rotmat77"],
        np.broadcast_to(np.eye(3, dtype=np.float32), result["rotmat77"].shape),
    )
    np.testing.assert_allclose(result["joints77"][:, 0], result["transl"])


def test_full_pose_batch_decode_preserves_input_order() -> None:
    bridge = object.__new__(FrozenMotionTokenizer)

    def fake_decode_batch(codes_list):
        return [np.full((1, 499), int(codes[0, 0]), dtype=np.float32) for codes in codes_list]

    def fake_materialize(features, **_kwargs):
        return {"marker": np.array([features[0, 0]], dtype=np.float32)}

    bridge.decode_batch = fake_decode_batch  # type: ignore[method-assign]
    bridge._materialize_full_pose_arrays = fake_materialize  # type: ignore[method-assign]
    codes = [
        np.array([[11]], dtype=np.int64),
        np.array([[22], [22]], dtype=np.int64),
        np.array([[33]], dtype=np.int64),
    ]
    anchors = [_anchor(), _anchor(), _anchor()]
    identities = [np.zeros(10, dtype=np.float32) for _ in codes]

    result = bridge.decode_to_full_pose_arrays_batch(
        codes, anchors, identities, device="cpu", batch_size=8,
    )

    assert [int(item["marker"][0]) for item in result] == [11, 22, 33]


def test_semoco_conversion_carries_full_pose_on_normal_soma_clip() -> None:
    frames = 2
    rotations = np.broadcast_to(
        np.eye(3, dtype=np.float32), (frames, 77, 3, 3),
    ).copy()
    translation = np.zeros((frames, 3), dtype=np.float32)
    joints = np.zeros((frames, 77, 3), dtype=np.float32)

    class _Tokenizer:
        def decode_to_full_pose_arrays_batch(self, *_args, **_kwargs):
            return [{
                "rotmat77": rotations,
                "transl": translation,
                "joints77": joints,
                "identity_coeffs": np.zeros((1, 10), dtype=np.float32),
                "foot_contacts": np.zeros((frames, 4), dtype=np.float32),
            }]

    anchor = {
        **_anchor(),
        "identity_coeffs": np.zeros(10, dtype=np.float32),
    }
    context = ConversionContext(
        device="cpu",
        semoco_tokenizer=_Tokenizer(),
        semoco_anchor=anchor,
    )
    source = MotionClip(
        rep="motion_codes",
        array=np.zeros((1, 16), dtype=np.int64),
        fps=50.0,
    )

    converted = build_default_graph().convert_batch([source], "soma77", context)[0]
    pose = SomaPose.from_motion_clip(converted, model_id="semoco")

    np.testing.assert_array_equal(pose.rotmat77, rotations)
    np.testing.assert_array_equal(pose.transl, translation)
    np.testing.assert_array_equal(pose.joints77, joints)
    assert pose.identity_coeffs is not None
    assert pose.foot_contacts is not None
