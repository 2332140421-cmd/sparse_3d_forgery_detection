"""Copy the frozen DeepAction media into one offline, unscored review bundle."""
import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

DERIVED = Path('/root/autodl-tmp/data/sparse_3d_forgery_detection/derived')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def ordered(rows):
    return sorted(rows, key=lambda r: hashlib.sha256(f"20260909:{r['video_id']}".encode()).hexdigest())


def training_examples(rows):
    selected, missing = [], []
    for name, folder in [('Wan', '02_wan'), ('FCVG', '03_fcvg')]:
        candidates = sorted([r for r in rows if r['split'] == 'train' and r['role'] == 'fake'
                             and folder in Path(r['video_path']).parts], key=lambda r: (r['source_id'], r['video_id']))
        if not candidates:
            missing.append(f'{name}: no frozen training fake identity'); continue
        fake = candidates[0]
        real = [r for r in rows if r['split'] == 'train' and r['role'] == 'real' and r['source_id'] == fake['source_id']]
        if len(real) != 1:
            missing.append(f'{name}/{fake["source_id"]}: real identity ambiguous or absent'); continue
        for r in [real[0], fake]:
            selected.append({**r, 'generator': name if r['role'] == 'fake' else 'Charades original',
                             'prompt': '训练数据比较示例', 'comparison_method': name,
                             'official_intervals': '未从本轮直接读取的冻结清单取得可靠时间区间；不按文件名推断'})
    return selected, missing


def validate_test(rows):
    if len(rows) != 40 or len({r['video_id'] for r in rows}) != 40 or len({r['action_id'] for r in rows}) != 20:
        raise ValueError('FROZEN_SET_COUNT_MISMATCH')
    if Counter(r['role'] for r in rows) != {'real': 20, 'fake': 20}:
        raise ValueError('ROLE_COUNT_MISMATCH')
    if Counter(r['generator'] for r in rows if r['role'] == 'fake') != {'RunwayML': 10, 'CogVideoX5B': 10}:
        raise ValueError('GENERATOR_COUNT_MISMATCH')
    if sum(r['metadata_frame_count'] for r in rows) != 12302:
        raise ValueError('FRAME_COUNT_MISMATCH')


def build(manifest, training, out):
    rows = json.loads(manifest.read_text())['rows']; validate_test(rows)
    examples, missing = training_examples(json.loads(training.read_text())['rows']) if training.is_file() else ([], ['training manifest unavailable'])
    mapping_root = DERIVED.parent/'datasets/v7_core_candidates/activityforensics_charades_v1'
    mapping_path = mapping_root/'paired/manifests/activityforensics_source_mapping.json'
    mapping = json.loads(mapping_path.read_text()) if mapping_path.is_file() else []
    for r in examples:
        if r['role'] != 'fake': continue
        matches = [m for m in mapping if m['charades_source_id'] == r['source_id']
                   and str(mapping_root/'source/activityforensics/raw'/m['activityforensics_file']) == r['video_path']]
        if len(matches) == 1 and matches[0].get('lineage_status') == 'EXACT':
            r['official_intervals'] = matches[0]['manipulation_intervals']
            r['annotation_evidence'] = matches[0]['lineage_evidence']
            r['annotation_mapping_sha256'] = sha(mapping_path)
    numbered = [{**r, 'review_id': f'V{i:03}', 'section': 'test'} for i, r in enumerate(sorted(rows, key=lambda r: r['video_id']), 1)]
    numbered += [{**r, 'review_id': f'T{i:03}', 'section': 'training_comparison'} for i, r in enumerate(examples, 1)]
    out.mkdir(parents=True, exist_ok=True)
    # Verify all original identities before copying any media. Never substitute.
    issues = []
    for r in numbered:
        p = Path(r['video_path'])
        if not p.is_file(): issues.append({'review_id': r['review_id'], 'path': str(p), 'error': 'MISSING'})
        elif sha(p) != r['video_sha256']: issues.append({'review_id': r['review_id'], 'path': str(p), 'error': 'SHA_MISMATCH'})
    if issues:
        (out/'identity_issues.json').write_text(json.dumps(issues, indent=2))
        raise ValueError('IDENTITY_BLOCKED: see identity_issues.json; no replacement/download')
    import av  # Existing media dependency; headers only, no frame decoding.
    for r in numbered:
        r['local_path'] = f"videos/{r['review_id']}.mp4"
        dst = out/r['local_path']; dst.parent.mkdir(exist_ok=True)
        if dst.exists():
            if dst.is_symlink() or sha(dst) != r['video_sha256']: raise ValueError(f'OUTPUT_CONFLICT: {dst}')
        else:
            shutil.copyfile(r['video_path'], dst)
        if sha(dst) != r['video_sha256']: raise ValueError(f'COPY_SHA_MISMATCH: {dst}')
        with av.open(str(dst)) as c:
            v = c.streams.video[0]
            r['video_codec'] = v.codec_context.name
            r['pixel_format'] = str(v.codec_context.format.name) if v.codec_context.format else None
        r['original_bytes'] = dst.stat().st_size
        r['preview_path'] = None
    package_id = hashlib.sha256(''.join(r['video_sha256'] for r in numbered).encode()).hexdigest()
    payload = {'package_id': package_id, 'rows': ordered([r for r in numbered if r['section'] == 'test']) + examples_order(numbered)}
    (out/'review_manifest.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    keys = ['review_id','section','video_id','source_id','action_id','prompt','role','generator','video_path','local_path',
            'video_sha256','original_bytes','official_path','official_revision','fps','duration_s','source_hw',
            'metadata_frame_count','video_codec','pixel_format','comparison_method','official_intervals','preview_path']
    with (out/'review_manifest.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, keys, extrasaction='ignore'); w.writeheader(); w.writerows(payload['rows'])
    template = Path(__file__).with_name('manual_review.html').read_text()
    (out/'index.html').write_text(template.replace('/*PAYLOAD*/null', json.dumps(payload, ensure_ascii=False).replace('<', '\\u003c')))
    summary = {'status': 'MATERIALS_READY_HUMAN_REVIEW_PENDING', 'test_videos': 40, 'actions': 20, 'test_frames': 12302,
               'training_comparison_videos': len(examples), 'missing_comparisons': missing,
               'manifest_sha256': sha(manifest), 'training_manifest_sha256': sha(training) if training.is_file() else None,
               'original_bytes': sum(r['original_bytes'] for r in numbered),
               'test_original_bytes': sum(r['original_bytes'] for r in numbered if r['section'] == 'test'),
               'preview_bytes': 0, 'codec_counts': dict(Counter(r['video_codec'] for r in numbered)),
               'human_records_prefilled': False, 'selection_seed': 20260909}
    (out/'preparation_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    (out/'README.md').write_text(README)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def examples_order(rows):
    return [r for r in rows if r['section'] == 'training_comparison']


README = '''# 原视频人工观看包（尚待人工审核）

把整个目录（含videos）通过VSCode远程资源管理器“下载”或SFTP复制到Windows，直接双击index.html。
不要只下载HTML，不用ZIP。默认每次只载入一段、不自动播放，保持原比例和自然速度。
默认只显示审核编号/动作/进度；“显示官方来源”用于第二遍查看标签、生成器、原路径和媒体参数。
这不是密码学盲测：CSV/页面源码含出处，第一遍请勿主动查看。
填写后导出JSON或CSV并保存到本机；也可备份到本包records/（自行创建）。网页不会自动写服务器文件。
localStorage可能被浏览器禁用/清除，不能替代导出备份。支持导入两种导出格式，按编号、video_id及SHA核对。
“完整看过”只能由您勾选，页面不会因播放结束自动认定。所有人工判断初始为空。

训练比较示例与40条测试分开导航/计数。训练real/fake为原始视频及其派生版本；
DeepAction只有动作语义匹配，不保证同场景同人物。时间区间没有可靠直接记录时标为未取得，绝不推断空间mask。

## 备用：在线查看（不公开服务）

服务器执行：python3 -m http.server 8765 --bind 127.0.0.1 --directory <本观看包绝对路径>
VSCode SSH的Ports面板转发8765，在Windows浏览器打开http://127.0.0.1:8765/。
若8765已占用，选择其他空闲端口并同步转发；结束后Ctrl+C。不需要Nginx或长期后台服务。

## 播放兼容性

本包保留原文件，未批量转码。codec信息见来源面板；服务器SHA/头部检查不等于Windows浏览器播放验收。
若仅某个文件不能播放，请先记录playback_problem并告知审核编号；只对该文件制作“仅供观看”的H.264预览。
预览须保留完整时长、比例、PTS/自然速度，并记录原SHA→预览文件/SHA；不替换原视频、不进入模型。
本轮没有证实不能播放的文件，故预览数量/空间均为0。不要因网页打开就认定人工审核完成。

## 解释边界（一页内）

A 数据/身份错误：SHA错配、官方映射错误、损坏/截断，应列具体证据；SHA一致不等于视频内容没有剪辑。
B 场景不匹配：室外、频繁切镜、强相机运动等，需人工记录，不能依据检测分数判断。
C 任务分布不同：训练主要为局部时间片段生成/修改后嵌回；DeepAction为独立生成短视频，语义匹配不是同源对照。
D 方法能力不足：身份、标签、场景正确仍未识别，或seed差异大，不能默认归因于数据。
上述可同时存在，当前不认定唯一原因。未观察到异常不等于真实；人工观感不覆盖官方来源标签。
异常/毛刺/碎片/跟踪困难不是自动排除规则。原40视频分数、权重、统计、阈值和结果不变。
不计算人工筛选后的AUROC。随后任何按人工记录形成的子集分析只能称事后探索，不能替代原测试。
人工粗记录不构成经验证的帧级或像素级伪造真值。
'''


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--manifest', type=Path, default=DERIVED/'v8_deepaction_cross_dataset_v1/video_manifest.json')
    p.add_argument('--training', type=Path, default=DERIVED/'v8_full_video_training_budget5_v1/input_manifest.json')
    p.add_argument('--output', type=Path, default=DERIVED/'v8_deepaction_manual_review_v1')
    a = p.parse_args(); build(a.manifest, a.training, a.output)
