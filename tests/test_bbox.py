"""Tests for the pure bbox→mask conversion logic.

These tests target _build_bbox_masks directly. The ComfyNode adapter
(nodules/bbox.py) is intentionally not imported here, so the test file
itself runs with no ComfyUI runtime stub. Loading the package's
__init__.py (required to access nodes._bbox_mask_logic via the package
path) is done through the normal pytest import path; comfy_api and
comfy.model_management are stubbed by conftest.py and the root project
conftest.
"""
import torch

from ComfyUI_Junnan_Tools.nodes._bbox_mask_logic import _build_bbox_masks


class TestBuildBboxMasks:
    def test_returns_zero_mask_when_no_detections(self):
        out = _build_bbox_masks([], batch_size=2, height=8, width=8)
        assert out.shape == (2, 8, 8)
        assert out.dtype == torch.float32
        assert (out == 0).all()

    def test_returns_zero_mask_for_frame_with_empty_list(self):
        out = _build_bbox_masks(
            [[], [{"x": 2, "y": 2, "width": 1, "height": 1}]],
            batch_size=2, height=4, width=4,
        )
        assert out.shape == (2, 4, 4)
        assert (out[0] == 0).all()
        assert out[1, 2, 2] == 1.0

    def test_merges_multiple_bboxes_into_single_frame_mask(self):
        bboxes = [[
            {"x": 0, "y": 0, "width": 2, "height": 2},
            {"x": 4, "y": 4, "width": 2, "height": 2},
        ]]
        out = _build_bbox_masks(bboxes, batch_size=1, height=8, width=8)
        assert out.shape == (1, 8, 8)
        assert (out[0, 0:2, 0:2] == 1).all()
        assert (out[0, 4:6, 4:6] == 1).all()
        assert out[0, 3, 3] == 0
        assert out[0, 0, 4] == 0

    def test_preserves_image_batch_to_mask_batch_alignment(self):
        bboxes = [
            [{"x": 0, "y": 0, "width": 1, "height": 1}],
            [],
            [{"x": 2, "y": 2, "width": 1, "height": 1}],
        ]
        out = _build_bbox_masks(bboxes, batch_size=3, height=4, width=4)
        assert out.shape == (3, 4, 4)
        assert out[0, 0, 0] == 1.0
        assert (out[1] == 0).all()
        assert out[2, 2, 2] == 1.0

    def test_broadcasts_single_bbox_frame_across_batch(self):
        bboxes = [[{"x": 1, "y": 1, "width": 1, "height": 1}]]
        out = _build_bbox_masks(bboxes, batch_size=3, height=4, width=4)
        assert out.shape == (3, 4, 4)
        assert torch.equal(out[0], out[1])
        assert torch.equal(out[1], out[2])
        assert out[0, 1, 1] == 1.0

    def test_clamps_bbox_to_image_bounds(self):
        bboxes = [[{"x": 6, "y": 6, "width": 100, "height": 100}]]
        out = _build_bbox_masks(bboxes, batch_size=1, height=8, width=8)
        assert out.shape == (1, 8, 8)
        assert (out[0, 6:8, 6:8] == 1).all()
        assert (out[0, :6, :] == 0).all()
        assert (out[0, :, :6] == 0).all()

    def test_skips_zero_area_bbox_after_clamp(self):
        bboxes = [[{"x": 10, "y": 10, "width": 0, "height": 0}]]
        out = _build_bbox_masks(bboxes, batch_size=1, height=4, width=4)
        assert out.shape == (1, 4, 4)
        assert (out == 0).all()
