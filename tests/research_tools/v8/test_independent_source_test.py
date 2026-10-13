import inspect

import numpy as np

from research_tools.v8.full_video_global_context.independent_test import GROUPS, metrics, used_sources
from research_tools.v8.full_video_global_context.runner import process


def test_preparatory_placeholder_is_not_performance_use():
    rows=[{'source_id':'planned','O_SET':''}, {'source_id':'missing','O_SET':'nan'},
          {'source_id':'scored','O_SET':'0'}, {'source_id':'measured','K0_S_norm':'1.25'}]
    assert used_sources(rows)=={'scored','measured'}


def test_fixed_threshold_video_metrics_and_three_groups():
    m=metrics(np.array([0,0,1,1]),np.array([-1.,0.,-2.,3.]))
    assert (m['tn'],m['fp'],m['fn'],m['tp'])==(1,1,1,1)
    assert m['fpr']==m['recall']==m['balanced_accuracy']==.5
    assert GROUPS==('COMPONENT_ADAPTED','COMPONENT_WINDOW','CELL_WINDOW')


def test_old_processor_defaults_preserved():
    parameters=inspect.signature(process).parameters
    assert parameters['total_videos'].default==272
    assert parameters['validate_first_block'].default is True
    assert parameters['model_timings'].default is False
