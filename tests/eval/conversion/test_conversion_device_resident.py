"""Tests for the direct device-resident joints22 -> soma77 conversion edge."""

from __future__ import annotations

import numpy as np
import torch

from semoco_generator.eval.conversions import ConversionContext, build_default_graph
from semoco_generator.eval.schema import MotionClip


class _FakeFitter:
    def __init__(self) -> None:
        self.calls: list[torch.Tensor] = []
        self.outputs: list[torch.Tensor] = []

    def joints22_to_vertices(self, joints):
        self.calls.append(joints)
        assert torch.is_tensor(joints)
        t = joints.shape[0]
        # deterministic fake "vertices": broadcast joints mean into a [T, 6890, 3]-like small stand-in
        out = joints.mean(dim=1, keepdim=True).expand(t, 5, 3).clone()
        self.outputs.append(out)
        return out


class _FakeSomaConverter:
    def __init__(self) -> None:
        self.received_tensors: list[torch.Tensor] = []

    def vertices_to_soma77(self, vertices):
        assert torch.is_tensor(vertices)
        self.received_tensors.append(vertices)
        t = vertices.shape[0]
        return np.asarray(vertices.detach().cpu().numpy()[:, :1, :].repeat(77, axis=1), dtype=np.float32).reshape(t, 77, 3)


def test_find_path_prefers_direct_joints22_to_soma77_edge():
    graph = build_default_graph()
    path = graph.find_path("joints22", "soma77")
    assert path == ["joints22", "soma77"]  # 1 hop, not via smpl_vertices
    print("find_path prefers direct joints22->soma77 edge OK")


def test_device_resident_edge_passes_gpu_tensor_without_cpu_roundtrip():
    graph = build_default_graph()
    ctx = ConversionContext(device="cpu")  # CPU is fine for this fake-based test
    fake_fitter = _FakeFitter()
    fake_soma = _FakeSomaConverter()
    ctx._fitter = fake_fitter
    ctx._soma = fake_soma

    clips = [
        MotionClip(rep="joints22", array=np.ones((4, 22, 3), dtype=np.float32), fps=30.0),
        MotionClip(rep="joints22", array=np.full((3, 22, 3), 2.0, dtype=np.float32), fps=30.0),
    ]

    results = graph.convert_batch(clips, "soma77", ctx)
    assert len(results) == 2
    for clip, orig in zip(results, clips):
        assert clip.rep == "soma77"
        assert clip.array.shape == (orig.array.shape[0], 77, 3)

    # Both clips fit within one frame budget chunk here, so exactly one fitter
    # call produced exactly one output tensor, and that exact object (not a
    # detach-and-reload copy) must be what the SOMA converter received — proof
    # there is no intervening .cpu().numpy() + re-upload round trip.
    assert len(fake_fitter.outputs) == 1
    assert len(fake_soma.received_tensors) == 1
    assert fake_soma.received_tensors[0] is fake_fitter.outputs[0]
    print("device-resident edge passes fitter output directly to soma converter OK")


def test_device_resident_edge_handles_frame_budget_chunking():
    graph = build_default_graph()
    ctx = ConversionContext(device="cpu")
    fake_fitter = _FakeFitter()
    fake_soma = _FakeSomaConverter()
    ctx._fitter = fake_fitter
    ctx._soma = fake_soma

    import semoco_generator.eval.conversions as conv_mod
    old_budget = conv_mod._MAX_FRAMES_PER_BATCH
    conv_mod._MAX_FRAMES_PER_BATCH = 3
    try:
        clips = [MotionClip(rep="joints22", array=np.ones((8, 22, 3), dtype=np.float32), fps=30.0)]
        results = graph.convert_batch(clips, "soma77", ctx)
    finally:
        conv_mod._MAX_FRAMES_PER_BATCH = old_budget

    assert len(results) == 1
    clip = results[0]
    assert clip.array.shape == (8, 77, 3)  # chunked internally (3+3+2) but reassembled correctly
    assert len(fake_fitter.calls) == 3  # 3 chunks
    print("device-resident edge frame-budget chunking OK")


if __name__ == "__main__":
    test_find_path_prefers_direct_joints22_to_soma77_edge()
    test_device_resident_edge_passes_gpu_tensor_without_cpu_roundtrip()
    test_device_resident_edge_handles_frame_budget_chunking()
    print("\nALL DEVICE-RESIDENT CONVERSION TESTS PASSED")
