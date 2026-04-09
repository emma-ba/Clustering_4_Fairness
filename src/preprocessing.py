"""
Preprocessing utilities for clustering fairness analysis.
"""

import pandas as pd


def encode_categoricals(df, col_lists, categorical_cols_arg, algorithm):
    """
    Encode categorical columns for clustering.

    For kprototypes: no encoding — returns the column names so per-condition
    indices can be computed later.
    For all other algorithms: one-hot encodes string/category columns and
    updates all column lists in-place.

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
            if dtype.kind in ('O', 'U', 'S') or dtype.name in ('category', 'string'):
                categorical_col_names.add(col)

    if algorithm == 'kprototypes':
        # kprototypes handles mixed types internally — no encoding needed
        return df, col_lists, list(categorical_col_names)

    if not categorical_col_names:
        return df, col_lists, []

    # One-hot encode each categorical column and update all col_lists
    df_encoded = df.copy()
    col_lists_updated = {name: list(cols) for name, cols in col_lists.items()}

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
        drop_first = (n_unique == 2)
        dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=drop_first).astype('int8')
        dummy_cols = list(dummies.columns)

        # Check for column name collisions with existing columns (excluding the col being replaced)
        existing_cols = set(df_encoded.columns) - {col}
        collisions = existing_cols & set(dummy_cols)
        if collisions:
            raise ValueError(
                f"One-hot encoding '{col}' would create columns {sorted(collisions)} that already "
                f"exist in the dataset. Rename or remove those columns before encoding."
            )

        df_encoded = pd.concat([df_encoded.drop(columns=[col]), dummies], axis=1)

        # Replace the original column with dummy columns in each list
        for cols in col_lists_updated.values():
            if col in cols:
                idx = cols.index(col)
                cols.remove(col)
                for j, dc in enumerate(dummy_cols):
                    cols.insert(idx + j, dc)

    return df_encoded, col_lists_updated, []
