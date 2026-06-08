# ──────────────────────────────────────────────────────────────────────────────
# Required packages:
#   pip install sdv imbalanced-learn scikit-learn numpy pandas matplotlib torch
#   pip install ForestDiffusion
#   pip install tabpfn            # TabPFN v2 — CPU-only flag built in
# SMOTE-NC requires ≥ 1 categorical column; for all-numeric data it falls back
# to regular SMOTE with a warning (behaviour is identical for pure-numeric input).
# ──────────────────────────────────────────────────────────────────────────────
"""
Expand a tabular CSV with synthetic generation + data augmentation.

Usage:
    python data_retrival/expand_dataset.py data_retrival/data/shapes/r1_c1.csv t
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

# ── optional heavy imports (graceful fallback) ────────────────────────────────
try:
    from sdv.single_table import (
        GaussianCopulaSynthesizer,
        TVAESynthesizer,
        CTGANSynthesizer,
        CopulaGANSynthesizer,
    )
    from sdv.metadata import SingleTableMetadata
    _SDV = True
except ImportError:
    _SDV = False
    warnings.warn("sdv not installed — SDV methods skipped.  pip install sdv")

try:
    from ForestDiffusion import ForestDiffusionModel
    _FOREST_DIFF = True
except ImportError:
    _FOREST_DIFF = False
    warnings.warn("ForestDiffusion not installed — diffusion model skipped.  pip install ForestDiffusion")

try:
    from tabpfn_client import TabPFNClassifier as _TabPFNClf
    try:
        from tabpfn_client import TabPFNRegressor as _TabPFNReg
    except ImportError:
        _TabPFNReg = None
    from tabpfn_client import set_access_token as _tabpfn_set_token
    _TABPFN = True
except ImportError:
    # fall back to local tabpfn if tabpfn-client is not installed
    try:
        from tabpfn import TabPFNClassifier as _TabPFNClf, TabPFNRegressor as _TabPFNReg
        _tabpfn_set_token = None
        _TABPFN = True
    except ImportError:
        _TABPFN = False
        warnings.warn("tabpfn / tabpfn-client not installed — TabPFN skipped.  pip install tabpfn-client")

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
    warnings.warn("imbalanced-learn not installed — SMOTE methods skipped.  pip install imbalanced-learn")


# ── CLI ───────────────────────────────────────────────────────────────────────
def _parse():
    p = argparse.ArgumentParser(description="Synthetic generation + augmentation for tabular CSVs.")
    p.add_argument("csv_path", type=Path, help="Input CSV file")
    p.add_argument("target",   type=str,  help="Target column name")
    p.add_argument("--bins",   type=int,  default=5,    help="Bins for discretising continuous target (SMOTE, default 5)")
    p.add_argument("--epochs", type=int,  default=300,  help="Training epochs for TVAE/CTGAN/CopulaGAN (default 300)")
    p.add_argument("--out",       type=Path, default=None, help="Output directory (default: same folder as input)")
    p.add_argument("--tabpfn-key", type=str,  default=None, help="TabPFN API key (overrides TABPFN_API_KEY env var)")
    return p.parse_args()


# ── helpers ───────────────────────────────────────────────────────────────────
def _discretize(series: pd.Series, n_bins: int) -> np.ndarray:
    """Bin a continuous series into integer class labels (quantile-based)."""
    n_bins = min(n_bins, len(series) // 2)          # never more bins than half the rows
    kbd = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
    return kbd.fit_transform(series.values.reshape(-1, 1)).ravel().astype(int)


def _safe(name: str, fn):
    """Run fn(); catch all exceptions, warn, return None so the pipeline continues."""
    try:
        return fn()
    except Exception as exc:
        warnings.warn(f"[{name}] failed: {exc}")
        return None


def _save(df: pd.DataFrame, path: Path, label: str):
    df.to_csv(path, index=False)
    print(f"  {label:30s}  {len(df):>6} rows  →  {path.name}")


# ── SDV synthetic generation ──────────────────────────────────────────────────
def _sdv_metadata(df: pd.DataFrame) -> "SingleTableMetadata":
    meta = SingleTableMetadata()
    meta.detect_from_dataframe(df)
    return meta


def _sdv_generate(df: pd.DataFrame, epochs: int) -> dict[str, pd.DataFrame | None]:
    """Fit each SDV synthesizer and sample len(df) rows."""
    if not _SDV:
        return {}

    n   = len(df)
    meta = _sdv_metadata(df)
    out  = {}

    # Gaussian Copula — fast, no neural net, stable at small n
    out["gaussian_copula"] = _safe(
        "GaussianCopula",
        lambda: GaussianCopulaSynthesizer(meta).fit(df) or
                GaussianCopulaSynthesizer(meta).fit_sample(df, n),  # API helper
    )
    # fit then sample (correct two-step API)
    def _fit_sample(cls, **kw):
        s = cls(meta, **kw)
        s.fit(df)
        return s.sample(num_rows=n)

    out["gaussian_copula"] = _safe("GaussianCopula", lambda: _fit_sample(GaussianCopulaSynthesizer))

    # TVAE — VAE-based; warn below 200 rows
    if n < 200:
        warnings.warn(f"TVAE: n={n} < 200 — training may be unstable.")
    out["tvae"] = _safe("TVAE", lambda: _fit_sample(TVAESynthesizer, epochs=epochs))

    # CTGAN — GAN-based; warn below 200 rows
    if n < 200:
        warnings.warn(f"CTGAN: n={n} < 200 — training may be unstable.")
    out["ctgan"] = _safe("CTGAN", lambda: _fit_sample(CTGANSynthesizer, epochs=epochs))

    # CopulaGAN — TabDDPM substitute; same stability caveat as CTGAN
    if n < 200:
        warnings.warn(f"CopulaGAN (TabDDPM substitute): n={n} < 200 — training may be unstable.")
    out["copula_gan"] = _safe("CopulaGAN", lambda: _fit_sample(CopulaGANSynthesizer, epochs=epochs))

    return out


# ── SMOTE-family augmentation ─────────────────────────────────────────────────
def _smote_resample(sampler, X: np.ndarray, y: np.ndarray, cols: list[str]) -> pd.DataFrame | None:
    """Fit-resample and return a DataFrame with original column names."""
    X_r, _ = sampler.fit_resample(X, y)
    return pd.DataFrame(X_r, columns=cols)


def _augment_smote(df: pd.DataFrame, target: str, n_bins: int) -> dict[str, pd.DataFrame | None]:
    if not _IMBLEARN:
        return {}

    n      = len(df)
    cols   = list(df.columns)
    X      = df.values.astype(float)   # ALL columns (including target) as features
    y_disc = _discretize(df[target], n_bins)

    # minimum neighbours must be < smallest class size
    min_class = int(np.bincount(y_disc).min())
    k = max(1, min(5, min_class - 1))

    if min_class < 2:
        warnings.warn("SMOTE: at least one class has < 2 samples — all SMOTE variants skipped.")
        return {}
    if n < 20:
        warnings.warn(f"SMOTE variants: n={n} < 20 — results may be unreliable.")

    cat_cols = [i for i, col in enumerate(cols)
                if df[col].dtype == object or str(df[col].dtype) == "category"]
    out = {}

    # SMOTE-NC (falls back to SMOTE when no categorical columns exist)
    if cat_cols and _SMOTENC:
        sampler = SMOTENC(categorical_features=cat_cols, k_neighbors=k)
        out["smote_nc"] = _safe("SMOTE-NC", lambda: _smote_resample(sampler, X, y_disc, cols))
    else:
        if not cat_cols:
            warnings.warn("SMOTE-NC: no categorical columns found — using SMOTE instead.")
        sampler = SMOTE(k_neighbors=k)
        out["smote_nc"] = _safe("SMOTE (NC-fallback)", lambda: _smote_resample(sampler, X, y_disc, cols))

    # BorderlineSMOTE
    out["borderline_smote"] = _safe(
        "BorderlineSMOTE",
        lambda: _smote_resample(BorderlineSMOTE(k_neighbors=k), X, y_disc, cols),
    )

    # ADASYN
    if n < 30:
        warnings.warn(f"ADASYN: n={n} < 30 — density estimation may fail.")
    out["adasyn"] = _safe(
        "ADASYN",
        lambda: _smote_resample(ADASYN(n_neighbors=k), X, y_disc, cols),
    )

    return out


# ── numeric augmentation ──────────────────────────────────────────────────────
def _gaussian_noise(df: pd.DataFrame, scale: float = 0.01) -> pd.DataFrame:
    """Add Gaussian noise scaled to each column's std (numeric columns only)."""
    out = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        sigma = df[col].std() * scale
        out[col] = df[col] + np.random.normal(0.0, sigma, len(df))
    return out


def _forest_diffusion(df: pd.DataFrame) -> pd.DataFrame | None:
    """ForestDiffusion: score-based diffusion model for tabular data.
    Trains a forest-based score function and generates len(df) new rows."""
    if not _FOREST_DIFF:
        return None
    if len(df) < 20:
        warnings.warn("ForestDiffusion: n < 20 — results may be unreliable.")
    X = df.values.astype(float)
    model = ForestDiffusionModel(
        X,
        n_t=50,          # diffusion timesteps
        duplicate_K=100, # number of duplicated samples for score estimation
        diffusion_type="flow",
        n_jobs=-1,
    )
    X_gen = model.generate(batch_size=len(df))
    return pd.DataFrame(X_gen, columns=df.columns)


def _tabpfn_generate(df: pd.DataFrame, target: str, api_key: str = "") -> pd.DataFrame | None:
    """TabPFN v2: Prior-Fitted Network trained in-context on the data.
    Uses tabpfn-client (cloud) when available, falls back to local tabpfn.
    Token is read from the TABPFN_API_KEY environment variable."""
    if not _TABPFN:
        return None

    import os
    n = len(df)
    if n > 10_000:
        warnings.warn("TabPFN v2: designed for n ≤ 10 000 — may be slow.")

    # authenticate with cloud client if token function is available
    if _tabpfn_set_token is not None:
        if not api_key:
            warnings.warn("TabPFN v2: no API key found — pass --tabpfn-key YOUR_KEY or set TABPFN_API_KEY.")
            return None
        _tabpfn_set_token(api_key)

    feature_cols = [c for c in df.columns if c != target]
    X = df[feature_cols].values.astype(float)
    y = df[target].values

    # use regressor for continuous target, classifier otherwise
    is_continuous = pd.api.types.is_float_dtype(df[target])
    if is_continuous and _TabPFNReg is not None:
        model = _TabPFNReg()
    else:
        if is_continuous:
            warnings.warn("TabPFN: no regressor available — discretising target for classifier.")
            y = _discretize(df[target], n_bins=5)
        model = _TabPFNClf()

    # model 2: predict target t from all features (x, y)
    model.fit(X, y)

    if X.shape[1] >= 2:
        # model 1: predict last feature (y) from all preceding features (x)
        X_prev = X[:, :-1]   # x columns
        y_feat = X[:, -1]    # y column
        model_feat = (_TabPFNReg() if is_continuous and _TabPFNReg is not None
                      else _TabPFNClf())
        model_feat.fit(X_prev, y_feat)

        # step 1: jitter x synthetically
        idx   = np.random.randint(0, n, n)
        sigma = X_prev.std(axis=0).clip(min=1e-6) * 0.05
        X_prev_new = X_prev[idx] + np.random.normal(0.0, sigma, (n, X_prev.shape[1]))

        # step 2: predict y from synthetic x
        y_feat_new = model_feat.predict(X_prev_new).reshape(-1, 1)

        # step 3: combine x and predicted y → predict t
        X_new = np.hstack([X_prev_new, y_feat_new])
    else:
        # single feature: fall back to jitter only
        idx   = np.random.randint(0, n, n)
        sigma = X.std(axis=0).clip(min=1e-6) * 0.05
        X_new = X[idx] + np.random.normal(0.0, sigma, (n, X.shape[1]))

    # predict t from (x, y)
    y_new = model.predict(X_new)

    df_new = pd.DataFrame(X_new, columns=feature_cols)
    df_new[target] = y_new
    return df_new[df.columns]


def _mixup(df: pd.DataFrame, alpha: float = 0.2) -> pd.DataFrame:
    """Tabular Mixup: convex combinations of random row pairs."""
    n    = len(df)
    idx1 = np.random.randint(0, n, n)
    idx2 = np.random.randint(0, n, n)
    lam  = np.random.beta(alpha, alpha, n)
    out  = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        v = df[col].values
        out[col] = lam * v[idx1] + (1.0 - lam) * v[idx2]
    return out


# ── visualization ─────────────────────────────────────────────────────────────
def _visualize(datasets: dict[str, pd.DataFrame], orig: pd.DataFrame,
               x_col: str, y_col: str, t_col: str, out_path: Path):
    """Scatter grid, one subplot per dataset, coloured by t — same style as
    create_correlation_shapes.py (dark blue = low t, pale blue = high t)."""
    CMAP = mcolors.LinearSegmentedColormap.from_list(
        "dark_pale_blue", ["#08306b", "#c6dbef"]
    )

    all_sets = {"original": orig} | {k: v for k, v in datasets.items() if v is not None}
    n_plots  = len(all_sets)
    ncols    = min(4, n_plots)
    nrows    = (n_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), facecolor="white")
    axes_flat = np.array(axes).reshape(-1)

    for ax, (label, df) in zip(axes_flat, all_sets.items()):
        # pick x/y axes — prefer named columns, fall back to first two numeric
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        px = x_col if x_col in df.columns else (num_cols[0] if num_cols else None)
        py = y_col if y_col in df.columns else (num_cols[1] if len(num_cols) > 1 else None)
        pt = t_col if t_col in df.columns else None

        if px and py:
            c = df[pt] if pt else df[px] + df[py]
            ax.scatter(df[px], df[py], c=c, cmap=CMAP, s=3, linewidths=0, alpha=0.8)
            ax.set_xlim(df[px].min(), df[px].max())
            ax.set_ylim(df[py].min(), df[py].max())

        ax.set_title(label.replace("_", " "), fontsize=13, pad=5)
        ax.axis("off")

    for ax in axes_flat[n_plots:]:   # hide unused cells
        ax.set_visible(False)

    plt.tight_layout(pad=0.4, h_pad=0.8, w_pad=0.4)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\n  Visualization → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    args    = _parse()
    csv_in  = args.csv_path
    target  = args.target
    out_dir = args.out or csv_in.parent
    stem    = csv_in.stem

    # load
    df = pd.read_csv(csv_in)
    print(f"Loaded  {len(df)} rows × {df.shape[1]} cols  from  {csv_in}")

    if target not in df.columns:
        sys.exit(f"ERROR: target '{target}' not found. Columns: {list(df.columns)}")

    # infer axis column names for visualization
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    x_col = "x" if "x" in df.columns else (num_cols[0] if num_cols else "")
    y_col = "y" if "y" in df.columns else (num_cols[1] if len(num_cols) > 1 else "")

    results: dict[str, pd.DataFrame | None] = {}

    # ── SDV generation ────────────────────────────────────────────────────────
    print("\n── SDV synthetic generation ──────────────────────────────────────────")
    results.update(_sdv_generate(df, epochs=args.epochs))

    # ── SMOTE augmentation ────────────────────────────────────────────────────
    print("\n── SMOTE augmentation ────────────────────────────────────────────────")
    results.update(_augment_smote(df, target, args.bins))

    # ── Gaussian noise ────────────────────────────────────────────────────────
    print("\n── Gaussian noise injection ──────────────────────────────────────────")
    results["gaussian_noise"] = _safe("GaussianNoise", lambda: _gaussian_noise(df, scale=0.01))

    # ── Tabular Mixup ─────────────────────────────────────────────────────────
    print("\n── Tabular Mixup ─────────────────────────────────────────────────────")
    results["mixup"] = _safe("Mixup", lambda: _mixup(df, alpha=0.2))

    # ── ForestDiffusion ───────────────────────────────────────────────────────
    print("\n── ForestDiffusion (score-based diffusion model) ─────────────────────")
    results["forest_diffusion"] = _safe("ForestDiffusion", lambda: _forest_diffusion(df))

    # ── TabPFN v2 (CPU) ───────────────────────────────────────────────────────
    print("\n── TabPFN v2 — CPU (prior-fitted network) ────────────────────────────")
    tabpfn_key = args.tabpfn_key or os.environ.get("TABPFN_API_KEY", "")
    results["tabpfn"] = _safe("TabPFN v2", lambda: _tabpfn_generate(df, target, tabpfn_key))

    # ── save individual CSVs ──────────────────────────────────────────────────
    print("\n── Saving output files ───────────────────────────────────────────────")
    out_dir.mkdir(parents=True, exist_ok=True)

    synthetic_frames = []
    for name, df_out in results.items():
        if df_out is None:
            print(f"  {'[skipped] ' + name:30s}")
            continue
        _save(df_out, out_dir / f"{stem}_{name}.csv", name)
        synthetic_frames.append(df_out)

    # combined synthetic-only CSV
    if synthetic_frames:
        df_combined = pd.concat(synthetic_frames, ignore_index=True)
        _save(df_combined, out_dir / f"{stem}_all_synthetic.csv", "ALL SYNTHETIC")

    # ── row-count summary ─────────────────────────────────────────────────────
    print("\n── Row-count summary ─────────────────────────────────────────────────")
    print(f"  {'original':30s}  {len(df):>6} rows")
    for name, df_out in results.items():
        tag = f"{len(df_out):>6} rows" if df_out is not None else "  skipped"
        print(f"  {name:30s}  {tag}")

    # ── visualization ─────────────────────────────────────────────────────────
    _visualize(results, df, x_col, y_col, target,
               out_dir / f"{stem}_augmented_viz.png")


if __name__ == "__main__":
    main()
