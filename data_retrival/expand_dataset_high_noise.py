# ──────────────────────────────────────────────────────────────────────────────
# Required packages:
#   pip install imbalanced-learn scikit-learn numpy pandas matplotlib
# ──────────────────────────────────────────────────────────────────────────────
"""
Same augmentation pipeline as expand_dataset.py but with significantly
increased noise (NOISE_STD = 10) applied to SMOTE-NC, BorderlineSMOTE,
ADASYN and Gaussian noise injection.

Usage:
    python data_retrival/expand_dataset_high_noise.py data_retrival/data/shapes/r1_c1.csv t
"""

import os
import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.preprocessing import KBinsDiscretizer

# ── noise level ───────────────────────────────────────────────────────────────
NOISE_STD = 10   # absolute Gaussian noise added after each augmentation step

# ── optional imports ──────────────────────────────────────────────────────────
try:
    from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
    try:
        from imblearn.over_sampling import SMOTENC
        _SMOTENC = True
    except ImportError:
        _SMOTENC = False
    _IMBLEARN = True
except ImportError:
    _IMBLEARN = False
    warnings.warn("imbalanced-learn not installed.  pip install imbalanced-learn")


# ── CLI ───────────────────────────────────────────────────────────────────────
def _parse():
    p = argparse.ArgumentParser(description="High-noise augmentation pipeline.")
    p.add_argument("csv_path", type=Path, help="Input CSV file")
    p.add_argument("target",   type=str,  help="Target column name")
    p.add_argument("--bins",   type=int,  default=5,    help="Bins for discretising target (default 5)")
    p.add_argument("--noise",  type=float, default=NOISE_STD, help=f"Gaussian noise std (default {NOISE_STD})")
    p.add_argument("--out",    type=Path, default=None, help="Output directory")
    return p.parse_args()


# ── helpers ───────────────────────────────────────────────────────────────────
def _discretize(series: pd.Series, n_bins: int) -> np.ndarray:
    n_bins = min(n_bins, len(series) // 2)
    kbd = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
    return kbd.fit_transform(series.values.reshape(-1, 1)).ravel().astype(int)


def _safe(name: str, fn):
    try:
        return fn()
    except Exception as exc:
        warnings.warn(f"[{name}] failed: {exc}")
        return None


def _add_noise(df: pd.DataFrame, std: float) -> pd.DataFrame:
    """Add Gaussian noise with fixed std to all numeric columns."""
    out = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        out[col] = df[col] + np.random.normal(0.0, std, len(df))
    return out


def _save(df: pd.DataFrame, path: Path, label: str):
    df.to_csv(path, index=False)
    print(f"  {label:30s}  {len(df):>6} rows  ->  {path.name}")


# ── SMOTE family ──────────────────────────────────────────────────────────────
def _smote_resample(sampler, X: np.ndarray, y: np.ndarray,
                    cols: list, noise_std: float) -> pd.DataFrame:
    X_r, _ = sampler.fit_resample(X, y)
    df_r = pd.DataFrame(X_r, columns=cols)
    # add high noise after resampling
    return _add_noise(df_r, noise_std)


def _augment(df: pd.DataFrame, target: str,
             n_bins: int, noise_std: float) -> dict:
    if not _IMBLEARN:
        return {}

    cols   = list(df.columns)
    X      = df.values.astype(float)
    y_disc = _discretize(df[target], n_bins)

    min_class = int(np.bincount(y_disc).min())
    k = max(1, min(5, min_class - 1))

    if min_class < 2:
        warnings.warn("SMOTE: fewer than 2 samples per class — skipped.")
        return {}

    out = {}
    cat_cols = [i for i, col in enumerate(cols)
                if df[col].dtype == object or str(df[col].dtype) == "category"]

    # SMOTE-NC (falls back to SMOTE for all-numeric data)
    if cat_cols and _SMOTENC:
        sampler = SMOTENC(categorical_features=cat_cols, k_neighbors=k)
    else:
        if not cat_cols:
            warnings.warn("SMOTE-NC: no categorical columns — using SMOTE instead.")
        sampler = SMOTE(k_neighbors=k)
    out["smote_nc"] = _safe(
        "SMOTE-NC",
        lambda: _smote_resample(sampler, X, y_disc, cols, noise_std),
    )

    # BorderlineSMOTE
    out["borderline_smote"] = _safe(
        "BorderlineSMOTE",
        lambda: _smote_resample(
            BorderlineSMOTE(k_neighbors=k), X, y_disc, cols, noise_std
        ),
    )

    # ADASYN
    out["adasyn"] = _safe(
        "ADASYN",
        lambda: _smote_resample(
            ADASYN(n_neighbors=k), X, y_disc, cols, noise_std
        ),
    )

    # Gaussian noise only (no SMOTE)
    out["gaussian_noise"] = _safe(
        "GaussianNoise",
        lambda: _add_noise(df, noise_std),
    )

    return out


# ── visualization ─────────────────────────────────────────────────────────────
def _visualize(results: dict, df_orig: pd.DataFrame,
               x_col: str, y_col: str, t_col: str,
               noise_std: float, out_path: Path):
    CMAP = mcolors.LinearSegmentedColormap.from_list(
        "dark_pale_blue", ["#08306b", "#c6dbef"]
    )
    all_sets = {"original": df_orig} | {k: v for k, v in results.items() if v is not None}
    n     = len(all_sets)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 5.5 * nrows),
                             facecolor="white")
    axes_flat = np.array(axes).reshape(-1)

    for ax, (label, df) in zip(axes_flat, all_sets.items()):
        num = df.select_dtypes(include=[np.number]).columns.tolist()
        px  = x_col if x_col in df.columns else (num[0] if num else None)
        py  = y_col if y_col in df.columns else (num[1] if len(num) > 1 else None)
        pt  = t_col if t_col in df.columns else None
        if px and py:
            c = df[pt] if pt else df[px] + df[py]
            ax.scatter(df[px], df[py], c=c, cmap=CMAP, s=4, linewidths=0, alpha=0.8)
        ax.set_title(label.replace("_", " "), fontsize=24, pad=10)
        ax.axis("off")

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle(f"noise std = {noise_std}", fontsize=14, y=1.01)
    plt.tight_layout(pad=1.5, h_pad=2.5, w_pad=1.5)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\n  PNG  ->  {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    args      = _parse()
    csv_in    = args.csv_path
    target    = args.target
    noise_std = args.noise
    out_dir   = args.out or csv_in.parent
    stem      = csv_in.stem

    df = pd.read_csv(csv_in)
    print(f"Loaded  {len(df)} rows x {df.shape[1]} cols  from  {csv_in}")
    print(f"Noise std = {noise_std}\n")

    if target not in df.columns:
        sys.exit(f"ERROR: target '{target}' not found. Columns: {list(df.columns)}")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    x_col = "x" if "x" in df.columns else (num_cols[0] if num_cols else "")
    y_col = "y" if "y" in df.columns else (num_cols[1] if len(num_cols) > 1 else "")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = _augment(df, target, args.bins, noise_std)

    # save CSVs
    print("-- Saving CSVs --")
    for name, df_out in results.items():
        if df_out is None:
            print(f"  [skipped] {name}")
            continue
        _save(df_out, out_dir / f"{stem}_hn_{name}.csv", name)

    # row-count summary
    print("\n-- Row-count summary --")
    print(f"  {'original':30s}  {len(df):>6} rows")
    for name, df_out in results.items():
        tag = f"{len(df_out):>6} rows" if df_out is not None else "  skipped"
        print(f"  {name:30s}  {tag}")

    # visualization
    _visualize(results, df, x_col, y_col, target, noise_std,
               out_dir / f"{stem}_hn_viz.png")


if __name__ == "__main__":
    main()
