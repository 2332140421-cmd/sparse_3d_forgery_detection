"""CPU-only asset gate for the requested adjacent-layer-swap pairs.

CLI caller; writes evidence and a blocked review page when no aligned object
alpha is available. Union edit GT is never silently converted to object alpha.
"""
import os
import subprocess
import time
from scipy.io import whosmat
from scripts.prepare_geometry_screen import ROOT, save, read
from scripts.prepare_v8_assets import digest


def eligible(rgb_complete, object_alpha_aligned, clean_background_verified):
    return rgb_complete and object_alpha_aligned and clean_background_verified


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    start = time.monotonic()
    out = ROOT / 'synthetic_temporal_pairs_v1'
    out.mkdir(exist_ok=True)
    if (out / 'summary.json').exists():
        raise RuntimeError('Existing result preserved; inspect before rerunning')
    records = []
    for num in ['04', '10', '02', '05', '06', '14']:
        mask = ROOT / 'grip' / f'GT_rigid_{num}.mat'
        records.append({'source': 'GRIP:' + num, 'mask_sha256': digest(mask),
                        'mat_fields': whosmat(mask), 'eligible': False,
                        'reason': 'Official binary GT is source/clone union, not isolated object alpha. No verified single-object alpha or independently verified clean background layer.'})
    archives = ROOT / 'ainpaint' / 'archives'
    records.append({'source': 'AINPAINT selected validation pool', 'eligible': False,
                    'complete_input_frames_archive': (archives / 'input_frames.tar').exists(),
                    'complete_input_masks_archive': (archives / 'input_masks.tar').exists(),
                    'reason': 'Original RGB archive incomplete; removal-region masks are not verified object alpha.'})
    assert not eligible(False, False, False)
    batch_path = ROOT / 'temporal_controls_v1' / 'human_review_batch.json'
    batch = read(batch_path)
    save(out / 'human_review_batch.json', batch)
    summary = {'status': 'ASSET_BLOCKED', 'process_exit_code': 0,
               'execution_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
               'passed_pairs': 0, 'constructed_pairs': 0, 'pair_checks': 'NOT_RUN_NO_ELIGIBLE_INPUT',
               't': None, 'H': None, 'planned_only': {'swap_indices': [4, 5], 't': 6, 'H': 3, 'unperturbed_t': 3},
               'records': records, 'human_batch_source_sha256': digest(batch_path),
               'pending': ['Verified moving-object RGB plus aligned alpha and clean background', 'Human four-item IDs and which item moved slightly'],
               'downloads': 0, 'model_calls': 0, 'depth_calls': 0, 'seconds': time.monotonic() - start}
    save(out / 'summary.json', summary)
    report = '''# 相邻对象图层交换：素材预检阻塞\n\n本轮构造0组、通过0组；未执行配对像素验收，不是配对验收失败。\n\n实读6个GRIP官方MAT字段均仅GT，结合已核实作者定义，仍为复制源／克隆并集。不能把并集当成独立运动对象alpha，也不能用差分、连通域或肉眼拆出目标。固定背景候选不等于已有核实干净背景层。AINPAINT仅mask包完整，原片包仍为part/range，不作完整RGB读取。没有补下载或生成替代对象。\n\n因此没有创建假合格的无损输入、目标mask或同步视频。t/H没有实际取值；仅登记未来计划：交换4/5、t=6、H=3，扰动前对照t=3。后续需先取得允许使用、逐帧对齐的运动对象RGB/alpha及干净背景，才能执行置换、历史哈希多重集与空间差异包含检查。\n\n原GRIP三组仍只作定性预览。人工四条自然编辑批次原样复制，3固定/1轻微移动、均协调难辨；ID及轻微移动对应项仍未知，未补逐条属性。\n\n本轮仅CPU元数据和既有记录核对，没有解码新全队列、特征、深度、模型或训练。程序退出0只表示预检报告生成成功；数据状态ASSET_BLOCKED。\n'''
    (out / 'report.md').write_text(report)
    (out / 'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>时序合成配对：素材待补</title><h1>0组：素材预检阻塞</h1><p>未构造顺序/扰动视频。GRIP并集GT不能替代对象alpha；AINPAINT原RGB不完整。不能人工确认不存在的合成配对。</p><p>t=6/H=3、交换4/5、对照t=3均仅是未执行计划。</p><p><a href="report.md">报告</a> / <a href="summary.json">逐来源证据</a> / <a href="human_review_batch.json">自然编辑人工批次</a></p><p><a href="../temporal_controls_v1/index.html">既有GRIP三组定性预览（不是本轮严格配对）</a></p><p>待补：对齐对象RGB/alpha和干净背景；人工四条ID及轻微移动对应项。</p>''')
    print(summary['status'], summary['seconds'])


if __name__ == '__main__':
    main()
