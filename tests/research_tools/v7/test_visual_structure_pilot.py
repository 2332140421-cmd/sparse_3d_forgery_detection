from __future__ import annotations

import numpy as np
import torch

from research_tools.v7.visual_structure_pilot.runner import (
    FusionModel,
    VisualHead,
    _preprocess_frame,
    _select_frames,
)
from research_tools.v7.geometry_information_pilot.model import ModalSetModel


def test_bin_center_selection_stays_inside_window_and_records_sixteen_times() -> None:
    row = {
        "window_id": "x",
        "interval_start_s": 10.0,
        "interval_end_s": 11.0,
        "parent_frame_indices": list(range(10, 30)),
        "parent_timestamps_s": [10.0 + i * 0.05 for i in range(20)],
    }
    selected = _select_frames(row)
    assert len(selected) == 16
    assert all(10.0 <= item["pts_s"] <= 11.0 for item in selected)
    assert all(10.0 <= item["target_s"] <= 11.0 for item in selected)


def test_preprocess_preserves_full_frame_with_mean_padding() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    out = _preprocess_frame(image, [0.5, 0.4, 0.3], [0.2, 0.2, 0.2], torch_module=torch)
    assert tuple(out.shape) == (3, 224, 224)
    # The letterbox borders are the normalized zero padding by construction.
    assert torch.allclose(out[:, 0, 0], torch.zeros(3), atol=1e-6)


def test_fusion_output_is_exact_sum_of_two_branches() -> None:
    model = FusionModel(ModalSetModel(8), VisualHead())
    states = torch.randn(3, 5, 8)
    intervals = torch.randn(3, 4)
    window_index = torch.tensor([0, 0, 1])
    visual = torch.randn(2, 512)
    fused, structural, visual_logits = model(states, intervals, window_index, 2, visual)
    assert torch.allclose(fused, structural + visual_logits, atol=1e-7, rtol=0)
