"""Tests for the MotionBatch device-resident conversion representation."""

from __future__ import annotations

import numpy as np
import torch

from semoco_generator.eval.motion_batch import MotionBatch, concat_batches
from semoco_generator.eval.schema import MotionClip


def _clip(n_frames: int, seed: int, *, with_aux: bool = False) -> MotionClip:
    rng = np.random.default_rng(seed)
    arr = rng.normal(size=(n_frames, 22, 3)).astype(np.float32)
    aux = {}
    if with_aux:
        aux["transl"] = rng.normal(size=(n_frames, 3)).astype(np.float32)
    return MotionClip(rep="joints22", array=arr, fps=30.0, aux=aux)


def test_from_clips_and_split_roundtrip():
    clips = [_clip(5, 0), _clip(3, 1), _clip(7, 2)]
    batch = MotionBatch.from_clips(clips, device="cpu")
    assert batch.data.shape[0] == 5 + 3 + 7
    assert batch.lengths == [5, 3, 7]

    out = batch.split()
    assert len(out) == 3
    for orig, got in zip(clips, out):
        assert got.rep == orig.rep
        np.testing.assert_allclose(got.array, orig.array, atol=1e-6)
        assert abs(got.fps - orig.fps) < 1e-6
    print("MotionBatch from_clips/split roundtrip OK")


def test_from_clips_stacks_aux_arrays():
    clips = [_clip(4, 0, with_aux=True), _clip(6, 1, with_aux=True)]
    batch = MotionBatch.from_clips(clips, device="cpu")
    assert "transl" in batch.aux
    assert batch.aux["transl"].shape[0] == 10
    out = batch.split()
    np.testing.assert_allclose(out[0].aux["transl"], clips[0].aux["transl"], atol=1e-6)
    np.testing.assert_allclose(out[1].aux["transl"], clips[1].aux["transl"], atol=1e-6)
    print("MotionBatch aux stacking roundtrip OK")


def test_to_device_is_noop_when_already_there():
    clips = [_clip(4, 0)]
    batch = MotionBatch.from_clips(clips, device="cpu")
    same = batch.to_device("cpu")
    assert same is batch  # no-op fast path returns the same object
    print("to_device no-op when already on target device OK")


def test_to_device_moves_tensors():
    clips = [_clip(4, 0)]
    batch = MotionBatch.from_clips(clips, device="cpu")
    moved = batch.to_device("meta")  # 'meta' device always available, no real GPU needed
    assert moved is not batch
    assert str(moved.data.device) == "meta"
    print("to_device moves tensors OK")


def test_slice_by_frame_budget_splits_without_losing_data():
    clips = [_clip(10, 0)]
    batch = MotionBatch.from_clips(clips, device="cpu")
    chunks = batch.slice_by_frame_budget(4)
    assert len(chunks) == 3  # 4 + 4 + 2
    assert sum(c.data.shape[0] for c in chunks) == 10
    recombined = concat_batches(chunks)
    assert recombined.data.shape[0] == 10
    torch.testing.assert_close(recombined.data, batch.data)
    print("slice_by_frame_budget + concat_batches roundtrip OK")


def test_slice_by_frame_budget_returns_self_when_under_budget():
    clips = [_clip(4, 0)]
    batch = MotionBatch.from_clips(clips, device="cpu")
    chunks = batch.slice_by_frame_budget(100)
    assert chunks == [batch]
    print("slice_by_frame_budget no-op under budget OK")


def test_estimate_bytes_matches_tensor_sizes():
    clips = [_clip(5, 0, with_aux=True)]
    batch = MotionBatch.from_clips(clips, device="cpu")
    expected = batch.data.numel() * batch.data.element_size()
    expected += sum(v.numel() * v.element_size() for v in batch.aux.values())
    assert batch.estimate_bytes() == expected
    print("estimate_bytes matches tensor sizes OK")


def test_concat_batches_reassembles_lengths_across_chunks_from_different_clips():
    clips = [_clip(3, 0), _clip(5, 1)]
    batch = MotionBatch.from_clips(clips, device="cpu")
    chunks = batch.slice_by_frame_budget(4)  # will NOT align with clip boundaries (3,4,1)
    combined = concat_batches(chunks)
    # Use the ORIGINAL batch.lengths (not chunk lengths) to split correctly.
    result = MotionBatch(rep=combined.rep, data=combined.data, lengths=batch.lengths, fps=combined.fps)
    out = result.split()
    assert len(out) == 2
    np.testing.assert_allclose(out[0].array, clips[0].array, atol=1e-6)
    np.testing.assert_allclose(out[1].array, clips[1].array, atol=1e-6)
    print("concat_batches reassembly across misaligned chunk boundaries OK")


def test_from_clips_is_actually_device_resident_on_real_cuda():
    """Real-GPU smoke test (skipped if no CUDA visible): confirms tensors are
    genuinely allocated on the CUDA device — not silently falling back to
    CPU — and that a full to_device -> slice -> concat -> split chain never
    touches host memory until ``split()``."""
    if not torch.cuda.is_available():
        print("test_from_clips_is_actually_device_resident_on_real_cuda SKIPPED (no CUDA)")
        return
    clips = [_clip(6, 0, with_aux=True), _clip(9, 1, with_aux=True)]
    batch = MotionBatch.from_clips(clips, device="cuda:0")
    assert batch.data.is_cuda
    assert batch.data.device.type == "cuda"
    assert all(v.is_cuda for v in batch.aux.values())

    moved = batch.to_device("cuda:0")
    assert moved is batch  # already there -> no-op fast path, still on GPU

    chunks = batch.slice_by_frame_budget(4)
    assert all(c.data.is_cuda for c in chunks)
    recombined = concat_batches(chunks)
    assert recombined.data.is_cuda
    torch.testing.assert_close(recombined.data, batch.data)

    out = batch.split()  # only here should data touch the host
    assert len(out) == 2
    np.testing.assert_allclose(out[0].array, clips[0].array, atol=1e-6)
    np.testing.assert_allclose(out[1].array, clips[1].array, atol=1e-6)
    free_b, total_b = torch.cuda.mem_get_info(torch.device("cuda:0"))
    print(
        f"MotionBatch real-CUDA device-residency OK "
        f"(device={torch.cuda.get_device_name(0)} free={free_b / (1 << 30):.1f}GiB/{total_b / (1 << 30):.1f}GiB)"
    )


if __name__ == "__main__":
    test_from_clips_and_split_roundtrip()
    test_from_clips_stacks_aux_arrays()
    test_to_device_is_noop_when_already_there()
    test_to_device_moves_tensors()
    test_slice_by_frame_budget_splits_without_losing_data()
    test_slice_by_frame_budget_returns_self_when_under_budget()
    test_estimate_bytes_matches_tensor_sizes()
    test_concat_batches_reassembles_lengths_across_chunks_from_different_clips()
    test_from_clips_is_actually_device_resident_on_real_cuda()
    print("\nALL MOTION BATCH TESTS PASSED")
