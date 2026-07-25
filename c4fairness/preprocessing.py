"""
Preprocessing utilities for clustering fairness analysis.
"""

import numpy as np
import pandas as pd


def encode_categoricals(
    df, col_lists, categorical_cols_arg, algorithm, distance="euclidean"
):
    """
    Encode categorical columns for clustering.

    For kprototypes: no encoding — returns the column names so per-condition
    indices can be computed later.
    For all other algorithms: one-hot encodes string/category columns and
    updates all column lists.

    Parameters
    ----------
    df : pd.DataFrame
    col_lists : dict of {name: list_of_cols}
        e.g. {'regular': [...], 'sensitive': [...], 'proxy': [...], 'special': [...]}
    categorical_cols_arg : list of str
        Explicitly specified categorical columns (e.g. from --categorical_cols).
    algorithm : str

    Returns
    -------
    df_encoded : pd.DataFrame
    col_lists_updated : dict
    categorical_col_names : list of str
        For kprototypes: names of categorical columns (caller computes indices per condition).
        For other algorithms: empty list (all columns are now numeric after encoding).
    multiclass_dummies : dict of {original_col: [dummy_col_names]}
        For each multi-class column that was encoded, maps the original column
        name to its dummy column names. Used downstream to build the readable
        fairness-analysis list (and, under Gower, to add dummies back since the
        col_lists there hold the factorized original).
    ohe_col_names : list of str
        Names of all one-hot encoded dummy columns (binary and multi-class).
        Empty for kprototypes and gower paths. Callers use this to exclude
        OHE dummies from StandardScaler.
    """
    # Collect all columns used in clustering
    seen = set()
    all_cols_ordered = []
    for cols in col_lists.values():
        for c in cols:
            if c not in seen:
                seen.add(c)
                all_cols_ordered.append(c)

    # Identify categorical columns: dtype-based + user-specified
    categorical_col_names = set(categorical_cols_arg or [])
    for col in all_cols_ordered:
        if col in df.columns:
            dtype = df[col].dtype
            if dtype.kind in ("O", "U", "S") or dtype.name in ("category", "string"):
                categorical_col_names.add(col)

    if algorithm == "kprototypes":
        return df, col_lists, list(categorical_col_names), {}, []

    if distance == "gower":
        df_encoded = df.copy()
        multiclass_dummies = {}
        for col in sorted(categorical_col_names):
            if col not in df_encoded.columns:
                continue
            dtype = df_encoded[col].dtype
            if not (
                dtype.kind in ("O", "U", "S") or dtype.name in ("category", "string")
            ):
                continue
            # Multi-class cols additionally get readable 0/1 dummies kept ALONGSIDE
            # the factorized original: the factorized code column stays in col_lists
            # for the Gower distance, the dummies feed the (readable) fairness
            # analysis (see _build_sensitive_analysis_list).
            if df_encoded[col].nunique(dropna=True) > 2:
                dummies = pd.get_dummies(
                    df_encoded[col], prefix=col, drop_first=False
                ).astype("int8")
                collisions = (set(df_encoded.columns) - {col}) & set(dummies.columns)
                if collisions:
                    raise ValueError(
                        f"One-hot encoding '{col}' would create columns {sorted(collisions)} that "
                        f"already exist in the dataset. Rename or remove those columns before encoding."
                    )
                multiclass_dummies[col] = list(dummies.columns)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
            # Factorize the original in place for the Gower distance.
            codes, _ = pd.factorize(df_encoded[col])
            codes = codes.astype(float)
            codes[codes < 0] = np.nan
            df_encoded[col] = codes
        return (
            df_encoded,
            col_lists,
            list(categorical_col_names),
            multiclass_dummies,
            [],
        )

    if not categorical_col_names:
        return df, col_lists, [], {}, []

    # One-hot encode each categorical column and update all col_lists
    df_encoded = df.copy()
    col_lists_updated = {name: list(cols) for name, cols in col_lists.items()}
    multiclass_dummies = {}
    all_ohe_dummy_cols = []

    for col in sorted(categorical_col_names):
        if col not in df_encoded.columns:
            continue

        n_unique = df_encoded[col].nunique(dropna=True)
        if n_unique <= 1:
            # Constant column — drop from all lists
            for cols in col_lists_updated.values():
                if col in cols:
                    cols.remove(col)
            continue

        # Binary: drop_first=True gives a single 0/1 column.
        # Multi-class: drop_first=False keeps all K categories.
        is_multiclass = n_unique > 2
        drop_first = not is_multiclass
        dummies = pd.get_dummies(
            df_encoded[col], prefix=col, drop_first=drop_first
        ).astype("int8")
        dummy_cols = list(dummies.columns)
        all_ohe_dummy_cols.extend(dummy_cols)

        # Check for column name collisions with existing columns (excluding the col being replaced)
        existing_cols = set(df_encoded.columns) - {col}
        collisions = existing_cols & set(dummy_cols)
        if collisions:
            raise ValueError(
                f"One-hot encoding '{col}' would create columns {sorted(collisions)} that already "
                f"exist in the dataset. Rename or remove those columns before encoding."
            )

        df_encoded = pd.concat([df_encoded.drop(columns=[col]), dummies], axis=1)

        if is_multiclass:
            multiclass_dummies[col] = dummy_cols

        # Update col_lists: replace the original column with its dummies in
        # every list that contained it (so they enter the feature matrix).
        for cols in col_lists_updated.values():
            if col not in cols:
                continue
            idx = cols.index(col)
            cols.remove(col)
            for j, dc in enumerate(dummy_cols):
                cols.insert(idx + j, dc)

    return df_encoded, col_lists_updated, [], multiclass_dummies, all_ohe_dummy_cols
