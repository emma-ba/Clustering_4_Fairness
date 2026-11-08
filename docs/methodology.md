# Methodology notes

Preprocessing and statistical details behind the results.

## Input data notes — one-hot encoding & StandardScaler

The pipeline auto-detects string/object/category columns and one-hot encodes them, keeping
the 0/1 dummies **out of `StandardScaler`**. Applying StandardScaler to binary OHE columns
distorts Euclidean distances — rarer categories get scaled to larger values and contribute
more to distances regardless of importance. Skipping it for OHE columns avoids this. See
[Cross Validated — bias when one-hot encoding and standardizing](https://stats.stackexchange.com/questions/612809/bias-towards-categorical-data-when-one-hot-encoding-and-standardizing-for-machi).

If your CSV already has externally one-hot-encoded 0/1 columns (integers, so not detected by
dtype), pass them via `--categorical_cols` so they are excluded from scaling — or use
`--no_standardize` if all features are already on comparable scales.

### kprototypes silhouette

Standard silhouette can't be computed directly for kprototypes (mixed numeric + categorical
distance). The pipeline precomputes the full pairwise distance matrix with the same distance
the algorithm uses (squared Euclidean + Hamming, weighted by the fitted gamma) and passes it
to `silhouette_score(metric='precomputed')`.

---

## R (optional) — exact multi-categorical Fisher

The omnibus significance test for multi-class errors and multi-categorical sensitive
features can use R's `fisher.test` (exact r×c Fisher–Freeman–Halton) via `rpy2`. This is
**optional**: without R, `c4fairness` falls back to scipy automatically (chi-square with
0-cell handling, or one-vs-all 2×2 Fisher). Install the extra + a system R ≥ 4.5 only if you
need the exact test on sparse tables (`--multicat_sig auto`/`fisher_rxc`):

```bash
pip install "c4fairness[r]"
```

On macOS the bundled R framework is often too old and `rpy2` picks it up, failing with
`symbol 'R_getVar' not found`. Install a modern R (`brew install r`) and point `rpy2` at it:

```bash
export R_HOME="$(/opt/homebrew/opt/r/bin/R RHOME)"   # Intel: /usr/local/opt/r/bin/R
export RPY2_CFFI_MODE=ABI
```
