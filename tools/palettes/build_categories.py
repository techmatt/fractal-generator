"""Build the durable palette COLOR-AXIS categorization -- one hierarchical tree
cut at k=8/12/16.

  uv run python tools/palettes/build_categories.py

Writes:
  * data/palettes/palette_categories.json  -- durable, load-bearing. Per palette:
    special in {neutral, spectral, outlier, chromatic}; chromatic gets a NESTED
    cluster id at each canonical k (8/12/16) + dendrogram leaf position; plus the
    full stored linkage so any k is derivable later. Default color axis = k=16.
  * out/palettes/palette_categories.html   -- disposable review sheet: the three
    cuts side-by-side with gradient swatches + fixed special sections.

Method (why hierarchical, not k-means-per-k): a SINGLE deterministic agglomerative
tree, cut at three heights, so every k=16 cell is a subset of exactly one k=12
cell, itself a subset of one k=8 cell -- coarse rolls up cleanly from fine, and
the reseed jitter that plagued k-means (~0.50 ARI floor) is gone. Independent
k-means per k would NOT nest; that is the whole reason for the tree.

Feature space is the resolved occupancy config frozen by the stability pass
(sigma=0.030 absolute OKLab, codebook=1024, PCA=20; see
scratchpad/palette_categories/stability.py and _resolved_config.json). Pre-pulls
(neutral / spectral / outlier) are reproduced deterministically and are FIXED
cells at every k; only the chromatic pool is clustered. Ward vs average linkage
are both built; the more balanced one (ward, by k=16 singleton count then size
std) is selected and recorded in the artifact.
"""
import json
import math
import os
import sys

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from scipy.spatial.distance import pdist

# durable artifact path (shared writer/reader contract)
ARTIFACT_PATH = 'data/palettes/palette_categories.json'
HTML_PATH = 'out/palettes/palette_categories.html'

ACHROMA_GATE = 0.030
SEED = 0
np.random.seed(SEED)

# resolved config (frozen by stability.py -> _resolved_config.json)
RES_SIGMA, RES_CB, RES_PCA = 0.030, 1024, 20
SPECTRAL_SPREAD = 0.82
OUTLIER_PCTL = 95
CAND_KS = [8, 12, 16]
DEFAULT_K = 16

# ---- load --------------------------------------------------------------------
feat = json.load(open('data/palettes/palette_features.json'))
pool = json.load(open('data/palettes/pool_colormaps.json'))
meta = {x['name']: x for x in pool}
names = list(feat.keys())
N = len(names)
T = np.array([feat[n]['trajectory'] for n in names])          # (N,32,3) OKLab

L_all = T[:, :, 0]
C_all = np.hypot(T[:, :, 1], T[:, :, 2])
H_all = np.arctan2(T[:, :, 2], T[:, :, 1]) % (2 * np.pi)
meanC = C_all.mean(1); meanL = L_all.mean(1)
_x = (C_all * np.cos(H_all)).sum(1); _y = (C_all * np.sin(H_all)).sum(1)
domHue = np.degrees(np.arctan2(_y, _x)) % 360
_R = np.hypot(_x, _y) / (C_all.sum(1) + 1e-12)
hue_spread = 1.0 - _R

# ---- neutral pre-pull (feature-independent low chroma) -----------------------
neutral_mask = meanC < ACHROMA_GATE
neutral = [names[i] for i in np.where(neutral_mask)[0]]
chrom_idx = np.where(~neutral_mask)[0]
print(f'N={N}  neutral={len(neutral)}  chromatic={len(chrom_idx)}')

# ---- occupancy feature (resolved codebook, cached nearest-color dist) ---------
ALL_COLORS = T.reshape(-1, 3)
km = MiniBatchKMeans(n_clusters=RES_CB, random_state=SEED, n_init=3,
                     batch_size=2048, max_iter=300).fit(ALL_COLORS)
cb = km.cluster_centers_

dmin2 = np.empty((len(chrom_idx), RES_CB))
for r, gi in enumerate(chrom_idx):
    d2 = ((cb[:, None, :] - T[gi][None, :, :]) ** 2).sum(-1)   # (kc,32)
    dmin2[r] = d2.min(1)
Fchrom = np.exp(-dmin2 / (2 * RES_SIGMA * RES_SIGMA))          # (nChrom, kc)
nChrom = Fchrom.shape[0]

# ---- spectral pre-pull (hue-spread) ------------------------------------------
loc_of = {names[chrom_idx[i]]: i for i in range(nChrom)}
spread_c = hue_spread[chrom_idx]
spectral_mask = spread_c > SPECTRAL_SPREAD
spectral = [names[chrom_idx[i]] for i in np.where(spectral_mask)[0]]

# ---- outlier pre-pull (occupancy-space NN percentile) ------------------------
rest_mask = ~spectral_mask
rest_local = np.where(rest_mask)[0]
Fn = Fchrom / (np.linalg.norm(Fchrom, axis=1, keepdims=True) + 1e-9)
Fr = Fn[rest_local]
G = Fr @ Fr.T
Dr = np.sqrt(np.maximum(0.0, 2 - 2 * G)); np.fill_diagonal(Dr, np.inf)
nn = Dr.min(1)
OUTLIER_NN = float(np.percentile(nn, OUTLIER_PCTL))
out_local_rest = np.where(nn > OUTLIER_NN)[0]
outlier = [names[chrom_idx[rest_local[i]]] for i in out_local_rest]

# ---- final clustering pool ---------------------------------------------------
pull_local = set(np.where(spectral_mask)[0]) | set(rest_local[out_local_rest])
pool_local = np.array([i for i in range(nChrom) if i not in pull_local])
pool_names = [names[chrom_idx[i]] for i in pool_local]
Fpool = Fchrom[pool_local]
nPool = len(pool_names)
print(f'pre-pulls: neutral={len(neutral)} spectral={len(spectral)} outlier={len(outlier)}'
      f' -> clustering pool={nPool}')
assert len(neutral) + len(spectral) + len(outlier) + nPool == N

# reduced representation: L2-normalize -> PCA-20
Fpn = Fpool / (np.linalg.norm(Fpool, axis=1, keepdims=True) + 1e-9)
Xpool = PCA(n_components=RES_PCA, random_state=SEED).fit_transform(Fpn)

# =============================================================================
# ONE TREE, CUT AT MANY HEIGHTS -- ward vs average, pick the more balanced
# =============================================================================
D = pdist(Xpool, metric='euclidean')
Z = {'ward': linkage(D, method='ward'),
     'average': linkage(D, method='average')}

def cut(Zm, k):
    """maxclust cut -> labels relabeled 1..K by dendrogram-leaf order."""
    lab = fcluster(Zm, t=k, criterion='maxclust')
    remap = {}
    for leaf in leaves_list(Zm):
        c = lab[leaf]
        if c not in remap:
            remap[c] = len(remap) + 1
    return np.array([remap[c] for c in lab]), int(len(remap))

def nesting_ok(fine, coarse):
    m = {}
    for f, c in zip(fine, coarse):
        if f in m and m[f] != c:
            return False, None
        m[f] = c
    return True, m

def size_stats(lab):
    sizes = np.bincount(lab)[1:]
    sizes = sizes[sizes > 0]
    return dict(k=len(sizes), sizes=sorted(sizes.tolist(), reverse=True),
                mn=int(sizes.min()), mx=int(sizes.max()),
                std=float(np.std(sizes)), singles=int((sizes == 1).sum()))

print('\n=== LINKAGE BALANCE (ward vs average) ===')
cuts, stats = {}, {}
for method in ('ward', 'average'):
    cuts[method] = {k: cut(Z[method], k) for k in CAND_KS}
    stats[method] = {}
    print(f'  {method}:')
    for k in CAND_KS:
        lab, ak = cuts[method][k]
        s = size_stats(lab); stats[method][k] = s
        print(f'    k={k:>2} (got {ak:>2}) sizes {s["sizes"]}  '
              f'min {s["mn"]}/max {s["mx"]}/std {s["std"]:.1f}  singletons {s["singles"]}')
    ok1, _ = nesting_ok(cuts[method][16][0], cuts[method][12][0])
    ok2, _ = nesting_ok(cuts[method][12][0], cuts[method][8][0])
    print(f'    nesting 16->12: {ok1}   12->8: {ok2}')
    assert ok1 and ok2, f'{method} cuts not nested (impossible on one tree)'

CHOSEN = min(('ward', 'average'), key=lambda m: (stats[m][16]['singles'], stats[m][16]['std']))
print(f'\n>>> CHOSEN linkage: {CHOSEN}  '
      f'(k=16 singletons ward={stats["ward"][16]["singles"]} avg={stats["average"][16]["singles"]}; '
      f'std ward={stats["ward"][16]["std"]:.1f} avg={stats["average"][16]["std"]:.1f})')

Zc = Z[CHOSEN]
leaf_order = leaves_list(Zc)
leaf_pos = {int(l): p for p, l in enumerate(leaf_order)}
cand_labels = {k: cuts[CHOSEN][k][0] for k in CAND_KS}

def rollup(fine_k, coarse_k):
    _, m = nesting_ok(cand_labels[fine_k], cand_labels[coarse_k])
    return {str(f): int(c) for f, c in sorted(m.items())}
nesting = {'16->12': rollup(16, 12), '12->8': rollup(12, 8)}

# =============================================================================
# DURABLE ARTIFACT
# =============================================================================
neutral_s, spectral_s, outlier_s = set(neutral), set(spectral), set(outlier)
palettes_map = {}
for n in names:
    if n in neutral_s:    sp = 'neutral'
    elif n in spectral_s: sp = 'spectral'
    elif n in outlier_s:  sp = 'outlier'
    else:                 sp = 'chromatic'
    entry = {'special': sp}
    if sp == 'chromatic':
        li = pool_names.index(n)
        entry['cluster'] = {str(k): int(cand_labels[k][li]) for k in CAND_KS}
        entry['leaf_pos'] = leaf_pos[li]
    else:
        entry['cluster'] = {str(k): sp for k in CAND_KS}
        entry['leaf_pos'] = None
    palettes_map[n] = entry

artifact = {
    'method': 'hierarchical-agglomerative',
    'linkage': CHOSEN,
    'metric': 'euclidean-on-pca20',
    'note': 'Color axis. One tree cut at k=8/12/16; nested by construction '
            '(k16 subset of k12 subset of k8). Specials (neutral/spectral/outlier) '
            'are fixed cells at every k. Default color axis = k=16. '
            'Generator: tools/palettes/build_categories.py.',
    'resolved': {'sigma': RES_SIGMA, 'codebook': RES_CB, 'pca': RES_PCA},
    'prepulls': {'neutral': len(neutral), 'spectral': len(spectral),
                 'outlier': len(outlier), 'spectral_spread_thresh': SPECTRAL_SPREAD,
                 'outlier_nn_pctl': OUTLIER_PCTL, 'outlier_nn_thresh': OUTLIER_NN},
    'canonical_ks': CAND_KS,
    'default_k': DEFAULT_K,
    'pool_size': nPool,
    'linkage_choice': {
        'chosen': CHOSEN,
        'balance': {m: {str(k): stats[m][k] for k in CAND_KS} for m in ('ward', 'average')},
    },
    'nesting': nesting,
    'pool_names': pool_names,
    'leaf_order_names': [pool_names[int(l)] for l in leaf_order],
    'linkage_matrix': Zc.tolist(),
    'palettes': palettes_map,
}
os.makedirs(os.path.dirname(ARTIFACT_PATH), exist_ok=True)
json.dump(artifact, open(ARTIFACT_PATH, 'w'), indent=1)
print(f'\nwrote {ARTIFACT_PATH}  (durable)')

# =============================================================================
# DISPOSABLE REVIEW SHEET (out/)
# =============================================================================
def full_map(k):
    m = {n: 0 for n in neutral}
    for n in spectral: m[n] = -2
    for n in outlier:  m[n] = -1
    lab = cand_labels[k]
    for li, n in enumerate(pool_names): m[n] = int(lab[li]) + 1
    return m
cand_maps = {k: full_map(k) for k in CAND_KS}

def buckets_of(label_map, cid_filter=lambda c: c > 0):
    b = {}
    for n, c in label_map.items():
        if cid_filter(c): b.setdefault(c, []).append(n)
    return b
def cluster_hue(members):
    x = sum(math.cos(math.radians(domHue[names.index(n)])) * meanC[names.index(n)] for n in members)
    y = sum(math.sin(math.radians(domHue[names.index(n)])) * meanC[names.index(n)] for n in members)
    return math.degrees(math.atan2(y, x)) % 360
def grad_css(stops):
    return 'linear-gradient(90deg,' + ','.join(
        f'rgb({r},{g},{b}) {round(t*100,2)}%' for t, (r, g, b) in stops) + ')'
def hue_swatch(hue):
    import colorsys
    r, g, b = colorsys.hls_to_rgb((hue % 360)/360, 0.5, 0.65)
    return f'rgb({int(r*255)},{int(g*255)},{int(b*255)})'
def pal_div(n):
    d = meta[n]
    return (f'<div class="pal" title="{n}  hue {domHue[names.index(n)]:.0f}  L {meanL[names.index(n)]:.2f}">'
            f'<div class="sw" style="background:{grad_css(d["stops"])}"></div>'
            f'<span class="pn">{n}</span></div>')
def special_section(title, members, sw_color):
    mm = sorted(members, key=lambda n: (domHue[names.index(n)], meanL[names.index(n)]))
    body = ''.join(pal_div(n) for n in mm)
    return (f'<section class="special"><div class="chead">'
            f'<span class="rep" style="background:{sw_color}"></span>'
            f'<h3>{title} <span class="n">n={len(members)}</span></h3></div>'
            f'<div class="grid">{body}</div></section>')
def variant_html(title, blurb, label_map):
    bl = sorted(buckets_of(label_map).values(), key=cluster_hue)
    sizes = sorted([len(m) for m in bl], reverse=True)
    parts = [f'<div class="vhead"><h2>{title}</h2><p class="blurb">{blurb}</p>'
             f'<p class="dist"><b>{len(bl)} clusters</b> &middot; sizes {sizes} &middot; '
             f'min {min(sizes)} / max {max(sizes)} / std {np.std(sizes):.1f}</p></div>']
    parts.append(special_section('NEUTRAL (achromatic)', neutral, '#888'))
    parts.append(special_section('SPECTRAL (full-coverage)', spectral,
                                 'conic-gradient(red,yellow,lime,cyan,blue,magenta,red)'))
    parts.append(special_section('OUTLIER (isolated)', outlier, '#444'))
    for m in bl:
        hue = cluster_hue(m)
        mm = sorted(m, key=lambda n: (domHue[names.index(n)], meanL[names.index(n)]))
        parts.append('<section><div class="chead">'
                     f'<span class="rep" style="background:{hue_swatch(hue)}"></span>'
                     f'<h3>hue {hue:.0f}&deg; <span class="n">n={len(m)}</span></h3>'
                     '</div><div class="grid">' + ''.join(pal_div(n) for n in mm) + '</div></section>')
    return '<div class="variant">' + '\n'.join(parts) + '</div>'

bal = ' | '.join(f'{m}: k16 std {stats[m][16]["std"]:.1f}/{stats[m][16]["singles"]} singles'
                 for m in ('ward', 'average'))
cols = ''.join(
    variant_html(f'k = {k}',
                 f'{CHOSEN} linkage &middot; one tree, maxclust cut'
                 + ('' if k == 16 else f' &middot; nested under k={16 if k == 8 else 16}'),
                 cand_maps[k])
    for k in CAND_KS)
html = ('<!doctype html><meta charset="utf-8"><title>Palette categories -- color axis</title>'
        '<style>'
        ':root{color-scheme:dark}'
        'body{background:#111;color:#ddd;font:13px/1.4 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:20px}'
        'h1{font-size:20px;margin:0 0 4px}.intro{color:#999;max-width:110ch;margin:0 0 8px}'
        '.kcurve{color:#cc9;font-family:ui-monospace,monospace;font-size:12px;margin:0 0 16px}'
        '.cols{display:flex;gap:16px;align-items:flex-start}'
        '.variant{flex:1 1 0;min-width:0;background:#0c0c0c;border:1px solid #2a2a2a;border-radius:8px;padding:12px}'
        '.vhead{position:sticky;top:0;background:#0c0c0c;padding:4px 0 8px;z-index:2;border-bottom:1px solid #333;margin-bottom:8px}'
        '.vhead h2{font-size:15px;margin:0 0 4px;color:#7bd}'
        '.blurb{color:#999;font-size:11px;margin:0 0 6px}'
        '.dist{color:#cc9;font-size:11px;margin:0;font-family:ui-monospace,monospace}.dist b{color:#fe8}'
        'section{margin:0 0 12px;border-top:1px solid #242424;padding-top:8px}'
        'section.special{background:#141414;border-radius:6px;padding:8px;border-top:2px solid #3a3a3a}'
        '.chead{display:flex;gap:8px;align-items:center;margin-bottom:6px}'
        '.rep{width:22px;height:22px;border-radius:5px;flex:0 0 auto;border:1px solid #000}'
        '.chead h3{font-size:13px;margin:0}.n{color:#7bd;font-weight:400}'
        '.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:5px}'
        '.pal{background:#171717;border-radius:4px;padding:3px;overflow:hidden}'
        '.sw{height:26px;border-radius:3px;border:1px solid #000}'
        '.pn{display:block;font-size:9px;color:#bbb;margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}'
        '</style>'
        '<h1>Palette color axis &mdash; hierarchical (one tree, cut at k=8/12/16)</h1>'
        f'<p class="intro">Resolved config <b>sigma={RES_SIGMA}, codebook={RES_CB}, PCA={RES_PCA}</b>. '
        'Single deterministic <b>agglomerative</b> tree (' + CHOSEN + ' linkage, euclidean on PCA-20), '
        'cut at three heights &mdash; every k=16 cell is a subset of exactly one k=12 cell, itself a subset '
        'of one k=8 cell (nesting verified). Pre-pulls fixed at every k: '
        f'<b>neutral {len(neutral)}</b>, <b>spectral {len(spectral)}</b>, <b>outlier {len(outlier)}</b>. '
        'Default color axis = <b>k=16</b>. Durable artifact: <code>' + ARTIFACT_PATH + '</code>.</p>'
        f'<p class="kcurve">linkage balance @k16 &mdash; {bal} &middot; chosen: <b>{CHOSEN}</b></p>'
        f'<div class="cols">{cols}</div>')
os.makedirs(os.path.dirname(HTML_PATH), exist_ok=True)
open(HTML_PATH, 'w', encoding='utf-8').write(html)
print(f'wrote {HTML_PATH}  (disposable review sheet)')
