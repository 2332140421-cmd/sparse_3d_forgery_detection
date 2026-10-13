import numpy as np
from matplotlib import colormaps
from matplotlib.colors import Normalize

from research_tools.v8.full_video_global_context.response_visualization import (
    GROUPS, SEEDS, model_axes, choose_times, mapped_heat,
)


def test_model_axis_is_read_from_saved_order_and_time_not_peak():
    models = list(reversed([f'{g}__seed{s}' for g in GROUPS for s in SEEDS]))
    axes = model_axes(models)
    assert [[models[i] for i in a] for a in axes] == [[f'{g}__seed{s}' for s in SEEDS] for g in GROUPS]
    j, interval, _ = choose_times(np.arange(9)*.25, '.5-1.5')
    assert j == [2, 4, 6] and interval == (.5, 1.5)
    assert choose_times(np.arange(9)*.25, '')[0] == [0, 4, 8]


def test_row_major_cells_partial_padding_and_crop():
    canvas = np.zeros((256, 256), bool); canvas[57:198, 19:236] = True
    content = canvas.reshape(16, 16, 16, 16).transpose(0, 2, 1, 3).reshape(256, 16, 16)
    mapping = {'pad_xy': [19, 57], 'content_hw': [141, 217]}
    heat = mapped_heat(np.arange(256), content, content.sum((1, 2)), mapping)
    assert heat.shape == (141, 217) and np.isfinite(heat).all()
    assert heat[0, 0] == 3*16+1
    assert heat[-1, -1] == 12*16+14


def test_absolute_negative_colors_do_not_use_per_image_minmax():
    values = np.array([-9., -6., -3.])
    normalized = Normalize(-19, 19)(values)
    assert np.all(normalized < .5)
    colors = colormaps['coolwarm'](normalized)
    assert np.all(colors[:, 2] > colors[:, 0])  # Blue-negative, not red-positive.
