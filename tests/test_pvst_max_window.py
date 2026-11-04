"""Protect only the one-window deadline, exact member accounting and local scope."""
import copy
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from research_tools.pvst.max_window_check import bounded_process,member_stats,summarize_hist


def test_timeout_terminates_only_owned_process_group(tmp_path):
    import subprocess
    unrelated=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)
    child_file=tmp_path/'child'
    script='import subprocess,sys,time; p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"]);open(sys.argv[1],"w").write(str(p.pid));time.sleep(30)'
    try:
        r=bounded_process([sys.executable,'-c',script,str(child_file)],.5,tmp_path/'worker.log')
        assert r['timed_out'] and r['returncode']<0 and r['termination_signal'] in [15,9] and r['wall_s']<7
        assert unrelated.poll() is None
        child=int(child_file.read_text());status=Path(f'/proc/{child}/status')
        assert not status.exists() or '\nState:\tZ' in status.read_text()
    finally:unrelated.terminate();unrelated.wait()


def test_completed_or_failed_exit_is_not_timeout(tmp_path):
    for code in [0,2,7]:
        r=bounded_process([sys.executable,'-c',f'raise SystemExit({code})'],2,tmp_path/'worker.log')
        assert r['returncode']==code and not r['timed_out'] and r['termination_signal'] is None


def test_membership_singletons_and_frame_sum_not_spatial_union():
    records=[{'fine':member_stats(np.array([1,1,3]))},{'fine':member_stats(np.array([1,2,2]))}]
    total=summarize_hist(records,'fine');assert total['nodes']==6 and total['points']==10 and total['singleton_fraction']==.5
    assert records[0]['fine']['count_quantiles_exact'][0]==1 and records[0]['fine']['count_quantiles_exact'][-1]==3


def test_sparse_candidate_requires_all_six_actual_layers():
    from types import SimpleNamespace
    import pytest
    from research_tools.pvst.max_window_check import candidate_layers,EXPECTED
    from research_tools.pvst.temporal import SparseLayer
    fine=[SparseLayer(256,4,512,8192) for _ in range(3)]
    coarse=[SparseLayer(512,8,512,8192) for _ in range(3)]
    model=SimpleNamespace(fine_time=SimpleNamespace(layers=fine),coarse_time=SimpleNamespace(layers=coarse))
    for l in fine+coarse:l.sparse_edge_attention=True;l.qkv_chunk=8192
    assert len(candidate_layers(model))==6
    with pytest.raises(ValueError,match='MISSING_ACTUAL'):candidate_layers(model,True)
    for layers,n in zip([fine,coarse],EXPECTED):
        for l,d in zip(layers,[1,2,4]):
            l.edge_stats={'nodes':n,'dilation':d,'completed_heads':l.heads,'head_impl':'research_tools.pvst.sparse_attention.sparse_head','backend':'CSR sampled_addmm/CSRSoftmax/CSRFromValues/CSR mm FP32 per head','softmax_impl':'research_tools.pvst.sparse_attention.CSRSoftmax','values_bridge':'research_tools.pvst.sparse_attention.CSRFromValues'}
    assert len(candidate_layers(model,True))==6
    coarse[-1].sparse_edge_attention=False
    with pytest.raises(ValueError,match='CANDIDATE_NOT_ENABLED'):candidate_layers(model)
