"""Exact media identities and nearest-neighbor semantic-mask coordinate mapping."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def media_identity(source: str, role: str, media_path: str) -> str:
    if not source or role not in {'real', 'fake'} or not media_path:
        raise ValueError('EXPLICIT_SOURCE_ROLE_PATH_REQUIRED')
    return hashlib.sha256(json.dumps([source, role, media_path], ensure_ascii=False).encode()).hexdigest()[:24]


def match_media(rows, source, role, media_path):
    matches = [r for r in rows if (r['source_id'], r['role'], r['media_path']) == (source, role, media_path)]
    if len(matches) != 1:
        raise ValueError(f'EXACT_MEDIA_MATCH_REQUIRED:{len(matches)}')
    return matches[0]


def content_mask(mapping):
    result = np.zeros(tuple(mapping['canvas_hw']), bool)
    left, top = mapping['pad_xy']; h, w = mapping['content_hw']
    result[top:top+h, left:left+w] = True
    return result


def mask_to_canvas(mask, mapping):
    if mask.shape != tuple(mapping['source_hw']) or mask.dtype != np.bool_:
        raise ValueError('SOURCE_BOOLEAN_MASK_REQUIRED')
    h, w = mapping['content_hw']; left, top = mapping['pad_xy']
    result = np.zeros(tuple(mapping['canvas_hw']), bool)
    result[top:top+h, left:left+w] = cv2.resize(mask.astype('uint8'), (w,h), interpolation=cv2.INTER_NEAREST).astype(bool)
    return result


def to_source(canvas, mapping, *, semantic_mask=False):
    if canvas.shape != tuple(mapping['canvas_hw']):
        raise ValueError('CANVAS_SHAPE')
    h,w = mapping['content_hw']; left,top = mapping['pad_xy']; sh,sw = mapping['source_hw']
    # Actual rounded width/height, rather than the nominal scalar, defines the inverse.
    crop = canvas[top:top+h, left:left+w]
    return cv2.resize(crop.astype('uint8') if semantic_mask else crop, (sw,sh),
                      interpolation=cv2.INTER_NEAREST if semantic_mask else cv2.INTER_LINEAR).astype(bool if semantic_mask else canvas.dtype)


def validate_annotation(row, root):
    status = row['annotation_status']
    if row['target'] != 'visible_forgery_anomaly':
        raise ValueError('ANNOTATION_TARGET')
    if status not in {'unannotated', 'uncertain', 'normal_empty', 'annotated'}:
        raise ValueError('ANNOTATION_STATUS')
    name = row.get('mask_path')
    if status in {'unannotated', 'uncertain'}:
        if name is not None: raise ValueError('UNREVIEWED_MASK')
        return None
    if not name: raise ValueError('REVIEWED_MASK_FILE_REQUIRED')
    mask = cv2.imread(str(Path(root)/name), cv2.IMREAD_UNCHANGED)
    if mask is None or mask.ndim != 2 or mask.shape != tuple(row['source_hw']) or not np.isin(mask,[0,255]).all():
        raise ValueError('ORIGINAL_SIZE_BINARY_PNG_REQUIRED')
    if (status == 'normal_empty') != (not mask.any()): raise ValueError('MASK_STATUS_CONFLICT')
    return mask > 0
