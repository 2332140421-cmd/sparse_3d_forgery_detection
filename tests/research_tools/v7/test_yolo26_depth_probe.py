import numpy as np
import pytest

from research_tools.v7.yolo26_depth_probe.compare import fixed_depth_limits, normalize_yolo_depth


class _Tensor:
    def __init__(self, value):
        self.data = value


class _Result:
    def __init__(self, value):
        self.depth = _Tensor(value)


def test_normalize_yolo_depth_accepts_singleton_tensor_dimension():
    result = _Result(np.ones((1, 2, 3), dtype=np.float32))
    output = normalize_yolo_depth(result, (2, 3))
    assert output.dtype == np.float32
    assert output.shape == (2, 3)


def test_normalize_yolo_depth_rejects_raster_mismatch():
    with pytest.raises(ValueError, match="does not match"):
        normalize_yolo_depth(_Result(np.ones((2, 2), dtype=np.float32)), (2, 3))


def test_fixed_depth_limits_are_shared_across_maps():
    limits = fixed_depth_limits([np.asarray([[1.0, 2.0]]), np.asarray([[3.0, 4.0]])])
    assert limits is not None
    assert limits == pytest.approx((1.06, 3.94))


def test_fixed_depth_limits_report_none_without_positive_finite_values():
    assert fixed_depth_limits([np.asarray([[np.nan, 0.0]])]) is None
