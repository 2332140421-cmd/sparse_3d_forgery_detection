"""Offline pilot audit: saved metrics, official masks and existing JPEG panels only.

CLI caller writes curves, per-video tables and a fixed-sample HTML index. No model,
checkpoint, frontend, media decoding or general cache loader is imported/called.
"""
import argparse
from collections import Counter
import csv
import hashlib
import html
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image, ImageDraw

from scripts.review_anakin import mask_array, numeric_members
from .data import mask_to_canvas

CONDITIONS = ('CELL_LOC', 'COMPONENT_LOC')
DATASETS = ('ActivityForensics', 'ANAKIN')


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def area_summary(fractions):
    """Native content fractions; AP's constant-score baseline uses positive frames only."""
    values = np.asarray(fractions, float)
    positive = values[values > 0]
    return {'positive_gt_frames': int(len(positive)), 'empty_gt_frames': int(len(values)-len(positive)),
            'area_median': float(np.median(positive)) if len(positive) else None,
            'area_min': float(positive.min()) if len(positive) else None,
            'area_max': float(positive.max()) if len(positive) else None,
            'ap_prevalence_baseline': float(positive.mean()) if len(positive) else None}


def lse_example(alpha, foreground=2., background=-2.):
    z = float(np.logaddexp(np.log(alpha)+foreground, np.log1p(-alpha)+background))
    required = float(np.log1p(-(1-alpha)*np.exp(background))-np.log(alpha))
    return {'alpha': alpha, 'foreground': foreground, 'background': background,
            'frame_logit': z, 'foreground_logit_for_frame_zero': required}


def stopping(history, min_delta=.002, patience=8):
    best = -np.inf; anchor = -np.inf; stale = 0; best_epoch = None
    for index, r in enumerate(history):
        dice = r['dev']['ANAKIN']['spatial_dice']
        if dice > best:
            best = dice; best_epoch = r['epoch']
        if dice > anchor+min_delta:
            anchor = dice; stale = 0
        else:
            stale += 1
        if stale >= patience and index != len(history)-1:
            raise ValueError('HISTORY_AFTER_EARLY_STOP')
    return {'best_epoch': best_epoch, 'stale': stale, 'patience_reached': stale >= patience}


def audit_masks(source, rows, destination):
    """Read only 32 official ZIPs and GT/label mmaps; never scan/hash large inputs."""
    areas = {}; checks = []
    for r in rows:
        if r['dataset'] != 'ANAKIN' or r['role'] != 'fake':
            continue
        directory = source/'cache'/r['video_id']
        gt = np.load(directory/'gt.npy', mmap_mode='r', allow_pickle=False)
        labels = np.load(directory/'frame_label.npy', mmap_mode='r', allow_pickle=False)
        known = np.load(directory/'pixel_known.npy', mmap_mode='r', allow_pickle=False)
        assert gt.shape == (r['actual_frames'], 512, 512)
        fractions = []; canvas_fractions = []; mismatches = 0
        with zipfile.ZipFile(r['mask_zip']) as z:
            members = numeric_members(z.namelist())
            assert members == [tuple(x) for x in r['mask_members']]
            assert [i for i, _ in members] == list(range(r['index_base'], r['index_base']+r['actual_frames']))
            for i, (_, name) in enumerate(members):
                binary = mask_array(z.read(name))[1]
                assert list(binary.shape) == r['source_hw']
                mapped = mask_to_canvas(binary, r['mapping512'])
                mismatches += int(not np.array_equal(mapped, gt[i]))
                assert known[i] and bool(labels[i]) == bool(binary.any())
                fractions.append(float(binary.mean()))
                canvas_fractions.append(float(mapped.sum()/np.prod(r['mapping512']['content_hw'])))
        assert mismatches == 0, 'GT_MAPPING_MISMATCH:'+r['video_id']
        areas[r['video_id']] = {**area_summary(fractions), 'fractions': fractions,
                               'canvas512': area_summary(canvas_fractions)}
        checks.append({'video_id': r['video_id'], 'pair_id': r['pair_id'], 'frames': len(fractions),
                       'gt_mapping_mismatches': mismatches, 'numeric_members_and_frame_labels': 'PASS'})
        print('mask audit', r['pair_id'], len(fractions), flush=True)
    save(destination/'mask_checks.json', checks)
    save(destination/'native_area_by_frame.json', areas)
    return areas


def video_table(rows, summaries, areas):
    result = []
    for condition, summary in summaries.items():
        for checkpoint in ('best', 'last'):
            available = {r['video_id']: r for key in (checkpoint+'_metrics', checkpoint+'_train_monitor')
                         for r in summary[key]['records']}
            for r in rows:
                if r['dataset'] != 'ANAKIN':
                    continue
                record = available.get(r['video_id'])
                stats = areas.get(r['video_id'], {})
                item = {k: r[k] for k in ('video_id', 'pair_id', 'source_id', 'split', 'role', 'generator', 'task', 'train_monitor')}
                item.update(condition=condition, checkpoint=checkpoint, epoch=summary[checkpoint+'_epoch'],
                            frames=r['actual_frames'], source_hw=str(r['source_hw']),
                            metric_status='saved' if record else 'not_evaluated_full_train',
                            **{k: stats.get(k) for k in ('positive_gt_frames', 'empty_gt_frames', 'area_median', 'area_min', 'area_max', 'ap_prevalence_baseline')})
                if r['role'] == 'real':
                    item.update(area_median=None, area_min=None, area_max=None,
                                positive_gt_frames=0, empty_gt_frames=r['actual_frames'])
                for k in ('spatial_dice', 'spatial_iou', 'pixel_ap', 'logit', 'empty_spatial_fpr'):
                    item[k] = record[k] if record else None
                cm = record.get('pixel_confusion_matrix') if record else None
                item['predicted_positive_pixel_fraction'] = (cm[0][1]+cm[1][1])/np.sum(cm) if cm else None
                item['pixel_ap_minus_prevalence'] = (item['pixel_ap']-item['ap_prevalence_baseline']
                    if item['pixel_ap'] is not None and item['ap_prevalence_baseline'] is not None else None)
                result.append(item)
    return result


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def curves(destination, histories):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for condition, history in histories.items():
        epochs = [r['epoch'] for r in history]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, part in zip(axes.flat, ('loss', 'video', 'frame', 'pixel')):
            ax.plot(epochs, [r['online_train'][part] for r in history], 'k--', label='online_train (changing weights)')
            key = 'loss' if part == 'loss' else 'loss_'+part
            for split in ('train_monitor', 'dev'):
                for d in DATASETS:
                    ax.plot(epochs, [r[split][d][key] for r in history], label=split+'/'+d)
            ax.set_title(part+' loss (frame plotted before x0.5)')
        finish(fig, axes, destination/(condition+'_losses.png'))
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        for split in ('train_monitor', 'dev'):
            for ax, metric in zip(axes[0, :2], ('spatial_dice', 'spatial_iou')):
                ax.plot(epochs, [r[split]['ANAKIN'][metric] for r in history], label=split+'/ANAKIN')
                ax.set_title(metric)
            for d in DATASETS:
                label = split+'/'+d
                for ax, task, metric in [(axes[0, 2], 'video', 'auroc'), (axes[1, 0], 'frame', 'auroc'), (axes[1, 1], 'frame', 'ap')]:
                    ax.plot(epochs, [r[split][d][task][metric] for r in history], label=label)
                    ax.set_title(task+' '+metric)
                axes[1, 2].plot(epochs, [r[split][d]['real_spatial_fpr'] for r in history], label=label)
        axes[1, 2].set_title('Real spatial FP fraction')
        finish(fig, axes, destination/(condition+'_tasks.png'))
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(epochs, [r['gradient_norm_mean'] for r in history], label='mean pre-clip norm; clip=1')
        axes[0].axhline(1, color='gray', linestyle=':'); axes[0].set_title('Pre-clip gradient norm (epoch mean)')
        axes[1].plot(epochs, [r['lr'] for r in history], label='learning rate'); axes[1].set_title('Learning rate')
        finish(fig, axes, destination/(condition+'_optimization.png'))


def finish(fig, axes, path):
    import matplotlib.pyplot as plt
    for ax in np.asarray(axes).flat:
        ax.set_xlabel('epoch'); ax.legend(fontsize=7); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def scatter(destination, table):
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    associations = {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, condition in zip(axes, CONDITIONS):
        associations[condition] = {}
        for split in ('train', 'dev'):
            rr = [r for r in table if r['condition'] == condition and r['checkpoint'] == 'best'
                  and r['split'] == split and r['role'] == 'fake' and r['spatial_dice'] is not None]
            x = [r['area_median'] for r in rr]; y = [r['spatial_dice'] for r in rr]
            ax.scatter(x, y, label='train_monitor n=4' if split == 'train' else 'dev n=8')
            for r in rr:
                ax.annotate(r['pair_id'][3:], (r['area_median'], r['spatial_dice']), fontsize=6)
            associations[condition][split] = {'n': len(rr), 'spearman_area_dice': float(spearmanr(x, y).statistic)}
        ax.set_xscale('log'); ax.set_xlabel('Native positive-frame median GT fraction (log scale)')
        ax.set_ylabel('Saved positive-frame/video-mean Dice'); ax.set_title(condition); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(destination/'area_dice.png', dpi=140); plt.close(fig)
    return associations


def sample_index(source, destination, rows):
    images = destination/'samples'; images.mkdir(exist_ok=True)
    selected = [r for r in rows if r['dataset'] == 'ANAKIN' and r['split'] == 'dev']
    entries = []; sheets = {'fake': [], 'real': []}
    for r in sorted(selected, key=lambda r: (r['role'], r['pair_id'])):
        time_data = [read(source/c/'samples'/(r['video_id']+'_time.json')) for c in CONDITIONS]
        assert time_data[0]['sample_indices'] == time_data[1]['sample_indices']
        assert time_data[0]['sample_indices'] == [int(r['actual_frames']*f) for f in (.25, .5, .75)]
        for j, fi in enumerate(time_data[0]['sample_indices']):
            originals = [Image.open(source/c/'samples'/f'{r["video_id"]}_{fi}.jpg').convert('RGB') for c in CONDITIONS]
            w, h = r['source_hw'][1], r['source_hw'][0]
            assert all(im.size == (3*w, h) for im in originals)
            # JPEG panels are display evidence only. Never threshold/count their pixels.
            panels = [originals[0].crop((0, 0, w, h)), originals[0].crop((w, 0, 2*w, h)),
                      originals[0].crop((2*w, 0, 3*w, h)), originals[1].crop((2*w, 0, 3*w, h))]
            header = 30; combined = Image.new('RGB', (4*w, h+header), 'white')
            draw = ImageDraw.Draw(combined)
            for k, (panel, title) in enumerate(zip(panels, ('Original (edited/real)', 'GT edit region', 'CELL_LOC binary', 'COMPONENT_LOC binary'))):
                combined.paste(panel, (k*w, header)); draw.text((k*w+5, 5), title, fill='black')
            name = f'{r["video_id"]}_{fi}.png'; combined.save(images/name)
            label = f'{r["pair_id"]} {r["role"]} {25+25*j}% frame={fi} PTS={time_data[0]["pts"][fi]:.4f}s'
            thumb = combined.copy(); thumb.thumbnail((1000, 240))
            strip = Image.new('RGB', (1000, thumb.height+25), 'white'); strip.paste(thumb, (0, 25))
            ImageDraw.Draw(strip).text((5, 4), label, fill='black'); sheets[r['role']].append(strip)
            entries.append({'video_id': r['video_id'], 'pair_id': r['pair_id'], 'role': r['role'],
                            'index': fi, 'fraction': [.25, .5, .75][j], 'pts_s': time_data[0]['pts'][fi],
                            'task': r['task'], 'image': 'samples/'+name,
                            'source_jpegs': [str(source/c/'samples'/f'{r["video_id"]}_{fi}.jpg') for c in CONDITIONS]})
    for role, strips in sheets.items():
        for part, start in enumerate(range(0, len(strips), 12)):
            batch = strips[start:start+12]; sheet = Image.new('RGB', (1000, sum(im.height for im in batch)), 'white'); y = 0
            for im in batch:
                sheet.paste(im, (0, y)); y += im.height
            sheet.save(destination/f'contact_{role}_{part+1}.png')
    body = ['<!doctype html><meta charset="utf-8"><title>V8固定开发样例</title>',
            '<style>body{font-family:sans-serif}img{max-width:100%}article{border-bottom:1px solid #aaa}</style>',
            '<h1>ANAKIN固定开发样例：8 fake及对应8 real，共48行</h1>',
            '<p>仅best模型已有JPEG二值预测，按25/50/75%原规则展示。原图为edited或real自身；不是预测概率图。',
            'JPEG上方原有文字保留；不从JPEG计数或重新评分。观察仅限三个固定时刻。</p>']
    for e in entries:
        body.append('<article><h3>'+html.escape(f'{e["pair_id"]} {e["role"]} {e["fraction"]:.0%} frame {e["index"]} PTS {e["pts_s"]:.4f}')+'</h3><p>'+html.escape(e['task'])+'</p><a href="'+e['image']+'"><img loading="lazy" src="'+e['image']+'"></a></article>')
    (destination/'index.html').write_text('\n'.join(body))
    save(destination/'sample_index.json', entries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); source = args.source; out = args.output
    if out.resolve() == source.resolve() or source.resolve() in out.resolve().parents:
        raise ValueError('DIAGNOSIS_MUST_BE_OUTSIDE_ORIGINAL_RUN')
    out.mkdir(parents=True, exist_ok=True)
    manifest = read(source/'joint_manifest.json'); rows = manifest['rows']
    summaries = {c: read(source/c/'summary.json') for c in CONDITIONS}
    histories = {c: read(source/c/'history.json') for c in CONDITIONS}
    checks = {}; inventory = {}
    for c, s in summaries.items():
        assert s['status'] == 'COMPLETE'
        for filename, expected in s['identity']['code_sha256'].items():
            assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == expected
        stop = stopping(histories[c]); assert stop['patience_reached'] and stop['best_epoch'] == s['best_epoch']
        assert len(histories[c]) == s['last_epoch']
        for checkpoint in ('best', 'last'):
            for subset in ('metrics', 'train_monitor'):
                scored = s[checkpoint+'_'+subset]
                for record in scored['records']:
                    scores = np.asarray(record['frame_scores'], float)
                    # Fixed per-video frame dimensions make area-normalized temporal
                    # LSE exactly log-mean-exp of the saved frame scores.
                    saved_video = float(np.logaddexp.reduce(scores)-np.log(len(scores)))
                    assert abs(saved_video-record['logit']) < 1e-10
                rr = [r for r in scored['records'] if r['dataset'] == 'ANAKIN' and r['positive_gt_frames']]
                for key in ('spatial_dice', 'spatial_iou'):
                    assert abs(float(np.mean([r[key] for r in rr]))-scored['datasets']['ANAKIN'][key]) < 1e-12
        inventory[c] = {'files': [p.name for p in (source/c).iterdir() if p.is_file()],
                        'sample_extensions': dict(Counter(p.suffix for p in (source/c/'samples').iterdir())),
                        'record_fields': list(s['best_metrics']['records'][0])}
        checks[c] = stop
    statistics = read(source/'statistics.json')
    assert statistics['train_video_ids'] == [r['video_id'] for r in rows if r['split'] == 'train']
    areas = audit_masks(source, rows, out)
    table = video_table(rows, summaries, areas); write_csv(out/'videos.csv', table)
    curves(out, histories); associations = scatter(out, table); sample_index(source, out, rows)
    math = [lse_example(a) for a in (.005, .01, .02, .05, .10)]
    save(out/'lse_examples.json', {'spatial': math, 'temporal_same_formula': 'replace alpha with positive-frame fraction when content sizes equal'})
    area_splits = {}
    for split in ('train', 'dev'):
        rr = [r for r in rows if r['dataset'] == 'ANAKIN' and r['role'] == 'fake' and r['split'] == split]
        medians = [areas[r['video_id']]['area_median'] for r in rr]
        area_splits[split] = {'videos': len(rr), 'edit_types': dict(Counter(r['generator'] for r in rr)),
            'frames': sum(r['actual_frames'] for r in rr),
            'positive_gt_frames': sum(areas[r['video_id']]['positive_gt_frames'] for r in rr),
            'median_video_area_medians': float(np.median(medians)),
            'min_video_area_median': min(medians), 'max_video_area_median': max(medians),
            'video_mean_positive_frame_mean_area': float(np.mean([areas[r['video_id']]['ap_prevalence_baseline'] for r in rr])),
            'video_mean_all_frame_mean_area': float(np.mean([np.mean(areas[r['video_id']]['fractions']) for r in rr]))}
    audit = {'source': str(source), 'checks': checks, 'inventory': inventory, 'associations': associations,
             'area_split_summary': area_splits,
             'area_scope': 'native official mask pixel fraction; positive-frame statistics; no padding; AP baseline mean over same positive frames then videos',
             'statistics_training_only': True, 'anakin_fake_videos': len(areas),
             'original_training_code_hashes_match': True,
             'cache_saved_fields_example': read(source/'cache'/rows[0]['video_id']/'complete.json')['fields'],
             'missing': ['raw pixel logits/probabilities', 'per-frame spatial scores/predicted areas',
                         'per-video metrics for 20 of 24 train fake videos', 'last checkpoint fixed pixel samples',
                         'separate pixel BCE/Dice loss logs', 'per-pair gradient norms in epoch history'],
             'input_small_files_sha256': {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in [source/'joint_manifest.json', source/'statistics.json', *[source/c/n for c in CONDITIONS for n in ('summary.json', 'history.json')]]}}
    save(out/'audit.json', audit)
    print('COMPLETE offline diagnosis artifacts:', out, flush=True)


if __name__ == '__main__':
    main()
