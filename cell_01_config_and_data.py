# 1. 读取比赛数据和训练配置
import os
import gc
import re
import json
import math
import copy
import random
import hashlib
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

DATA_ROOT = ''  # 自动查找失败时，手动填写detection目录
OUT = Path('/kaggle/working')
OUT.mkdir(parents=True, exist_ok=True)
SMOKE_TEST = False
CFG = dict(split_seed=2026, seeds=[2026, 4099, 8191], size=512, folds=5,
           epochs=80, batch=8, repeats=3, lr=4e-4, min_lr=5e-6,
           weight_decay=1e-3, ema=0.99, workers=0, heat_sigma=1.5,
           coarse_sigma=0.85, pixel_unit=32.0, pair_count=2,
           pair_supervision=0.25, js_weight=0.5, consistency_weight=0.1,
           tail_epochs=[60, 70, 80], checkpoint_every=5)
if SMOKE_TEST:
    CFG.update(seeds=[2026], size=128, folds=2, epochs=2, batch=2,
               repeats=1, pair_count=1, tail_epochs=[1, 2], checkpoint_every=1)
assert CFG['size'] % 32 == 0
assert CFG['tail_epochs'] == sorted(set(CFG['tail_epochs']))
assert CFG['tail_epochs'][-1] == CFG['epochs']
assert min(CFG['tail_epochs']) > 0 and len(set(CFG['seeds'])) == len(CFG['seeds'])
S, HS, CS = CFG['size'], CFG['size'] // 2, 16
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
AMP = DEVICE.type == 'cuda'
if not AMP and not SMOKE_TEST:
    raise RuntimeError('正式训练请先启用 Kaggle GPU；CPU 只建议 SMOKE_TEST。')
cv2.setNumThreads(0)
torch.set_num_threads(min(4, os.cpu_count() or 1))
torch.backends.cudnn.benchmark = AMP


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if AMP:
        torch.cuda.manual_seed_all(seed)


def discover_root():
    if DATA_ROOT:
        root = Path(DATA_ROOT)
        if not (root / 'fovea_localization_train_GT.csv').is_file():
            raise FileNotFoundError('DATA_ROOT 应填写 detection 目录')
        return root
    base = Path('/kaggle/input')
    files = list(base.rglob('fovea_localization_train_GT.csv'))
    if not files:
        for archive in sorted(base.rglob('*.zip')):
            if not zipfile.is_zipfile(archive):
                continue
            with zipfile.ZipFile(archive) as z:
                names = [n for n in z.namelist() if n.endswith('fovea_localization_train_GT.csv')]
                if not names:
                    continue
                if len(names) != 1:
                    raise RuntimeError('ZIP 内含多份训练数据，请设置 DATA_ROOT')
                prefix = names[0].rsplit('/', 1)[0] + '/' if '/' in names[0] else ''
                folder = OUT / 'context_data' / archive.stem
                for name in z.namelist():
                    if not name.startswith(prefix) or name.endswith('/'):
                        continue
                    target = (folder / name).resolve()
                    if folder.resolve() not in target.parents:
                        raise ValueError('ZIP 路径不安全')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(z.read(name))
                files.append(folder / names[0])
    if len(files) != 1:
        raise RuntimeError('找到 %d 份训练数据，请设置 DATA_ROOT。' % len(files))
    return files[0].parent


def numeric_id(value):
    text = str(value).strip()
    if re.fullmatch(r'\d+\.0+', text):
        return int(float(text))
    stem = Path(text).stem
    if not re.fullmatch(r'\d+', stem):
        raise ValueError('无法解析图片 ID: ' + text)
    return int(stem)


def image_index(folder):
    result = {}
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}:
            continue
        ident = numeric_id(p.name)
        if ident in result:
            raise ValueError('重复图片 ID: %d' % ident)
        result[ident] = p
    return result


def read_rgb(path):
    bgr = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError('无法读取图片: ' + str(path))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def transform_points(points, matrix):
    points = np.asarray(points, np.float32).reshape(-1, 2)
    return points @ matrix[:, :2].T + matrix[:, 2]


def resize_affine(x0, y0, width, height, new_width, new_height, pad_x, pad_y):
    # 缩放时按像素中心换算，避免坐标偏半个像素
    sx, sy = new_width / width, new_height / height
    return np.array([[sx, 0, pad_x - sx * x0 + (sx - 1) / 2],
                     [0, sy, pad_y - sy * y0 + (sy - 1) / 2]], np.float32)


def canvas_image(rgb):
    h, w = rgb.shape[:2]
    small = cv2.resize(rgb, (max(1, w // 8), max(1, h // 8)), interpolation=cv2.INTER_AREA)
    mask = small.max(axis=2) > 12
    ys, xs = np.flatnonzero(mask.mean(axis=1) > .08), np.flatnonzero(mask.mean(axis=0) > .08)
    if len(xs) and len(ys):
        x0 = max(0, int(xs[0] * w / small.shape[1]) - 8)
        x1 = min(w, int((xs[-1] + 1) * w / small.shape[1]) + 8)
        y0 = max(0, int(ys[0] * h / small.shape[0]) - 8)
        y1 = min(h, int((ys[-1] + 1) * h / small.shape[0]) + 8)
    else:
        x0, y0, x1, y1 = 0, 0, w, h
    width, height = x1 - x0, y1 - y0
    ratio = S / max(width, height)
    nw, nh = max(1, round(width * ratio)), max(1, round(height * ratio))
    px, py = (S - nw) // 2, (S - nh) // 2
    resized = cv2.resize(rgb[y0:y1, x0:x1], (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((S, S, 3), np.uint8)
    canvas[py:py + nh, px:px + nw] = resized
    return canvas, resize_affine(x0, y0, width, height, nw, nh, px, py)


def read_polygon(root, ident, point, visible):
    if not visible:
        return None
    xml = root / 'train_location' / ('%04d.xml' % ident)
    objects = [o for o in ET.parse(str(xml)).getroot().findall('object')
               if 'fovea' in (o.findtext('name') or '').lower()]
    if len(objects) != 1:
        raise ValueError('Fovea 框数目错误: ' + str(xml))
    box = objects[0].find('bndbox')
    x0, y0, x1, y1 = [float(box.findtext(k)) for k in ['xmin', 'ymin', 'xmax', 'ymax']]
    if not (x0 <= point[0] <= x1 and y0 <= point[1] <= y1):
        raise ValueError('训练坐标不在框内: %d' % ident)
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)


def cache_item(root, ident, path, point=None):
    rgb = read_rgb(path)
    canvas, matrix = canvas_image(rgb)
    h, w = rgb.shape[:2]
    item = dict(id=ident, image=canvas, inverse=cv2.invertAffineTransform(matrix),
                wh=np.array([w, h], np.float32))
    if point is not None:
        vis = bool(np.any(np.asarray(point) != 0))
        if vis and not (0 <= point[0] < w and 0 <= point[1] < h):
            raise ValueError('训练坐标超出图像: %d' % ident)
        polygon = read_polygon(root, ident, point, vis)
        item.update(visible=float(vis), xy=transform_points([point], matrix)[0] if vis else np.zeros(2, np.float32),
                    polygon=transform_points(polygon, matrix) if polygon is not None else None)
    return item


def make_folds(items, count, seed):
    groups = {}
    for i, item in enumerate(items):
        key = hashlib.sha256(item['image'].tobytes()).hexdigest()
        groups.setdefault(key, []).append(i)
    values = list(groups.values())
    np.random.RandomState(seed).shuffle(values)
    values.sort(key=len, reverse=True)
    result = [[] for _ in range(count)]
    for group in values:
        result[min(range(count), key=lambda f: len(result[f]))].extend(group)
    if not all(result):
        raise ValueError('重复图分组后不足以划分验证折')
    return [np.array(sorted(f), int) for f in result]


ROOT = discover_root()
gt = pd.read_csv(ROOT / 'fovea_localization_train_GT.csv')
gt.columns = gt.columns.str.strip()
assert {'data', 'Fovea_X', 'Fovea_Y'}.issubset(gt.columns)
gt['id'] = gt['data'].map(numeric_id)
gt = gt.sort_values('id').reset_index(drop=True)
coords = gt[['Fovea_X', 'Fovea_Y']].apply(pd.to_numeric).to_numpy(np.float32)
assert gt['id'].is_unique and np.isfinite(coords).all() and (coords >= 0).all()
visible = np.any(coords != 0, axis=1)
assert visible.any()
HAS_NEGATIVE = bool((~visible).any())
train_paths, test_paths = image_index(ROOT / 'train'), image_index(ROOT / 'test')
assert set(gt['id']) == set(train_paths)
sample = pd.read_csv(ROOT / 'sample_submission.csv', dtype={'ImageID': str})
sample.columns = sample.columns.str.strip()
assert list(sample.columns) == ['ImageID', 'value'] and sample['ImageID'].is_unique
expected_ids = {'%d_Fovea_%s' % (ident, axis) for ident in test_paths for axis in ['X', 'Y']}
assert set(sample['ImageID']) == expected_ids
train_items = [cache_item(ROOT, int(row.id), train_paths[int(row.id)], coords[i]) for i, row in gt.iterrows()]
test_items = [cache_item(ROOT, ident, path) for ident, path in sorted(test_paths.items())]
folds = make_folds(train_items, CFG['folds'], CFG['split_seed'])
print('Device=%s Train=%d Test=%d Seeds=%d Folds=%d' %
      (DEVICE, len(gt), len(test_items), len(CFG['seeds']), CFG['folds']), flush=True)
