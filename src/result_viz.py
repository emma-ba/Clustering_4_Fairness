"""
Result-table presentation: column classification, spec labels, and heatmaps.

Result tables (Overview + Detailed) are organised into three metric families,
each with its own color family (spec: Size = blue, Error = red, Sensitive =
violet). p-value / significance columns use the *reversed* colormap so that a
lower p (more significant) renders darker. Category columns (the "winning"
error type / sensitive category) carry text, not a magnitude, so they are
drawn as flat tinted cells.
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns


# Size-family columns (Overview + Detailed) -> spec display label.
_SIZE_COLS = {
    'silh': 'silh.', 'silhouette': 'silh.',
    'min_size': 'min size', 'min_prop': 'min prop.', 'max_prop': 'max prop.',
    'size': 'size', 'size_prop': 'size', 'count': 'size',
    'proportion': 'prop.', 'prop': 'prop.',
}

# Error-family columns keyed by exact name (the per-cluster magnitude columns).
# kind: 'value' | 'pvalue' | 'category'; '{e}' is replaced by the user error label.
_ERROR_EXACT = {
    'error': ('value', '{e}'),
    'error_value': ('value', '{e}'),
    'error_mean': ('value', '{e} mean'),
    'abs_error_mean': ('value', '|{e}| mean'),
}

# Error-family columns keyed by suffix on 'error...'. Longest suffix first so
# '_gap_sig' wins over '_gap'.
_ERROR_SUFFIXES = [
    ('_gap_sig', 'pvalue', '{e} gap sig.'),
    ('_gap_class', 'category', '{e} gap class'),
    ('_gap_cat', 'category', '{e} gap cat.'),
    ('_gap', 'value', '{e} gap'),
    ('_sep', 'pvalue', '{e} sep.'),
    ('_cat', 'category', '{e} cat.'),
    ('_value', 'value', '{e}'),
]

# Sensitive-feature columns keyed by suffix on '<F>...'. Longest suffix first.
_FEAT_SUFFIXES = [
    ('_gap_sig', 'pvalue', '{f} gap sig.'),
    ('_gap_cat', 'category', '{f} gap cat.'),
    ('_gap', 'value', '{f} gap'),
    ('_sep', 'pvalue', '{f} sep.'),
    ('_cat', 'category', '{f} cat.'),
    ('_value', 'value', '{f}'),
]

# (family, kind) -> matplotlib colormap. *_r reverses so lower p = darker.
_FAMILY_CMAP = {
    ('size', 'value'): 'Blues', ('size', 'pvalue'): 'Blues_r',
    ('error', 'value'): 'Reds', ('error', 'pvalue'): 'Reds_r',
    ('sensitive', 'value'): 'Purples', ('sensitive', 'pvalue'): 'Purples_r',
    ('meta', 'value'): 'Greys', ('meta', 'pvalue'): 'Greys_r',
}

# Flat background tint for category (text) cells, per family.
_FAMILY_TINT = {
    'size': '#dbe9f6', 'error': '#fcdbd5', 'sensitive': '#e7e1f2', 'meta': '#eeeeee',
}


def classify_column(col, error_label='error'):
  """Map a result-table column to (family, kind, display_label).

  family: 'size' | 'error' | 'sensitive' | 'meta'
  kind:   'value' | 'pvalue' | 'category'
  """
  if col in _SIZE_COLS:
    return 'size', 'value', _SIZE_COLS[col]
  if col in _ERROR_EXACT:
    kind, tmpl = _ERROR_EXACT[col]
    return 'error', kind, tmpl.format(e=error_label)
  if col.startswith('error='):
    # onehot per-class error columns: 'error=<class>[_gap|_gap_sig|_sep]'
    rest = col[len('error='):]
    for suf, kind, lbl in (('_gap_sig', 'pvalue', ' gap sig.'), ('_sep', 'pvalue', ' sep.'),
                           ('_gap', 'value', ' gap')):
      if rest.endswith(suf):
        return 'error', kind, f'{error_label}={rest[:-len(suf)]}{lbl}'
    return 'error', 'value', f'{error_label}={rest}'
  if col.startswith('error_'):
    rest = col[len('error'):]  # leading '_' kept so suffixes match
    for suf, kind, tmpl in _ERROR_SUFFIXES:
      if rest == suf:
        return 'error', kind, tmpl.format(e=error_label)
    return 'error', 'value', error_label
  for suf, kind, tmpl in _FEAT_SUFFIXES:
    if col.endswith(suf):
      return 'sensitive', kind, tmpl.format(f=col[:-len(suf)])
  return 'meta', 'value', col


def display_label(col, error_label='error'):
  """Spec display label for a result-table column (see classify_column)."""
  return classify_column(col, error_label)[2]


def order_result_columns(cols, error_label='error'):
  """Stable partition of columns into Size -> Error -> Sensitive -> meta order,
  preserving the input order within each family (callers emit per-feature columns
  contiguously, so this yields the spec column order)."""
  fams = {'size': [], 'error': [], 'sensitive': [], 'meta': []}
  for c in cols:
    fams[classify_column(c, error_label)[0]].append(c)
  return fams['size'] + fams['error'] + fams['sensitive'] + fams['meta']


def _annot_strings(df_segment):
  """Annotation array for a category (text) segment: NaN/None -> ''."""
  def to_str(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
      return ''
    return str(v)
  return np.vectorize(to_str, otypes=[object])(df_segment.values)


def render_result_heatmap(df, output_path, error_label='error', title=None,
                          figsize=None, fmt='.4g'):
  """Render a result table as a color-coded heatmap.

  Columns are coloured by family (Size=blue, Error=red, Sensitive=violet); p-value
  columns use the reversed colormap (lower = darker); category columns are drawn as
  flat tinted text cells. Adjacent columns sharing a colormap are merged into one
  sub-axes so the labels stay slanted and the families read as blocks. `df` columns
  are expected to already be in display order (see order_result_columns)."""
  cols = list(df.columns)
  if not cols:
    return

  specs = [classify_column(c, error_label) for c in cols]
  # Per-column color key: ('__cat__', family) for text cols, else (cmap, family).
  keys = [('__cat__', fam) if kind == 'category'
          else (_FAMILY_CMAP.get((fam, kind), 'Greys'), fam)
          for fam, kind, _ in specs]

  # Merge adjacent columns that share a color key into segments.
  segments = []
  for i, key in enumerate(keys):
    if segments and segments[-1][0] == key:
      segments[-1][1].append(i)
    else:
      segments.append([key, [i]])

  col_counts = [len(idxs) for _, idxs in segments]
  n_rows = len(df)
  if figsize is None:
    figsize = (max(6, len(cols) * 1.1), max(4, n_rows * 0.6))

  fig, axes = plt.subplots(1, len(segments), figsize=figsize,
                           gridspec_kw={'width_ratios': col_counts, 'wspace': 0})
  if len(segments) == 1:
    axes = [axes]

  for ax, ((cmap, fam), idxs) in zip(axes, segments):
    seg_cols = [cols[i] for i in idxs]
    sub = df[seg_cols]
    labels = [specs[i][2] for i in idxs]
    if cmap == '__cat__':
      bg = pd.DataFrame(0.0, index=sub.index, columns=sub.columns)
      sns.heatmap(bg, annot=_annot_strings(sub), fmt='', cbar=False, ax=ax,
                  cmap=mcolors.ListedColormap([_FAMILY_TINT[fam]]),
                  linewidths=0.5, linecolor='white')
    else:
      sns.heatmap(sub.astype(float), annot=True, fmt=fmt, cbar=False, ax=ax,
                  robust=True, cmap=cmap)
    ax.xaxis.tick_top()
    ax.tick_params(axis='x', which='major', length=0)
    ax.tick_params(axis='y', which='major', length=0)
    ax.set_xticklabels(labels, rotation=45, ha='left', rotation_mode='anchor')
    ax.set(xlabel='', ylabel='')

  for ax in axes[1:]:
    ax.set_yticklabels([])

  if title:
    fig.suptitle(title, y=1.02)
  plt.tight_layout()
  plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
  plt.close(fig)


def plot_quality_heatmap(all_quali_viz, output_path, figsize=None,
                         error_label='error', title=None):
  """Plot the Overview quality heatmap with blue/red/violet color families.

  Size metrics (silhouette, cluster sizes/proportions) are blue, error metrics
  red, sensitive-feature metrics violet; p-value columns render darker when more
  significant. Column names are mapped to spec labels and the user `error_label`."""
  df = all_quali_viz.copy()
  cols = order_result_columns(list(df.columns), error_label)
  if not cols:
    return
  render_result_heatmap(df[cols], output_path, error_label=error_label,
                        title=title, figsize=figsize)


def plot_cluster_recap_heatmap(recap, cond_name, output_dir, multiclass_dummies=None,
                               error_label='error'):
  """Plot the per-cluster Detailed heatmap (one row per cluster).

  Columns are ordered Size -> Error -> Sensitive and coloured by family
  (blue / red / violet) via the shared render_result_heatmap: value columns use
  the family colormap, p-value columns its reverse (lower = darker), and category
  columns render as flat tinted text. `multiclass_dummies` is accepted for
  call-site compatibility but no longer used (each feature is a single column)."""
  df = recap.copy()
  if 'c' in df.columns:
    df = df.set_index('c')
    df.index = [f'cluster {i}' for i in df.index]
  # n_error is a raw count duplicated by error_value (rate); not a display column.
  df = df.drop(columns=[col for col in ('n_error',) if col in df.columns])

  cols = order_result_columns(list(df.columns), error_label)
  if not cols:
    return
  out_path = f'{output_dir}/' + re.sub(' +', '', cond_name) + '.png'
  render_result_heatmap(df[cols], out_path, error_label=error_label,
                        title=re.sub(' +', ' ', cond_name))
