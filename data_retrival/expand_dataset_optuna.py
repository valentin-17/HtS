# ──────────────────────────────────────────────────────────────────────────────
# Required packages:
#   pip install optuna sdv imbalanced-learn scikit-learn scipy
#   pip install numpy pandas matplotlib torch ForestDiffusion tabpfn-client
# ──────────────────────────────────────────────────────────────────────────────
"""
Same pipeline as expand_dataset.py, but every generative model gets
20 Optuna trials to optimise its hyperparameters.
The best run per model is saved as CSV + JPG scatter plot.

Quality metric (objective to minimise):
    mean Kolmogorov-Smirnov statistic across all numeric columns
    between the real data and the synthetic data  (lower = more similar).

Usage:
    python data_retrival/expand_dataset_optuna.py data_retrival/data/shapes/r1_c1.csv t
    python data_retrival/expand_dataset_optuna.py data_retrival/data/shapes/r1_c1.csv t --trials 20 --tabpfn-key YOUR_KEY
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
from scipy.stats import ks_2samp
from sklearn.preprocessing import KBinsDiscretizer

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)   # silence per-trial noise

# ── optional imports (same graceful fallback as expand_dataset.py) ────────────
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
    warnings.warn("sdv not installed — SDV models skipped.  pip install sdv")

try:
    from ForestDiffusion import ForestDiffusionModel
    _FOREST = True
except ImportError:
    _FOREST = False
    warnings.warn("ForestDiffusion not installed — skipped.  pip install ForestDiffusion")

try:
    from tabpfn_client import TabPFNRegressor as _TabPFNReg, set_access_token as _tabpfn_auth
    _TABPFN = True
except ImportError:
    try:
        from tabpfn import TabPFNRegressor as _TabPFNReg
        _tabpfn_auth = None
        _TABPFN = True
    except ImportError:
        _TABPFN = False


# ── CLI ───────────────────────────────────────────────────────────────────────
def _parse():
    p = argparse.ArgumentParser(description="Optuna-tuned synthetic data generation.")
    p.add_argument("csv_path",     type=Path, help="Input CSV file")
    p.add_argument("target",       type=str,  help="Target column name")
    p.add_argument("--trials",     type=int,  default=20,   help="Optuna trials per model (default 20)")
    p.add_argument("--tabpfn-key", type=str,  default=None, help="TabPFN API key")
    p.add_argument("--out",        type=Path, default=None, help="Output directory")
    return p.parse_args()


# ── helpers ───────────────────────────────────────────────────────────────────
def _quality(df_real: pd.DataFrame, df_syn: pd.DataFrame) -> float:
    """Mean KS statistic across numeric columns — lower means more similar."""
    cols = [c for c in df_real.select_dtypes(include=[np.number]).columns
            if c in df_syn.columns]
    if not cols:
        return 1.0
    return float(np.mean([
        ks_2samp(df_real[c].dropna().values, df_syn[c].dropna().values)[0]
        for c in cols
    ]))


def _safe(name: str, fn):
    try:
        return fn()
    except Exception as exc:
        warnings.warn(f"[{name}] failed: {exc}")
        return None


def _sdv_meta(df: pd.DataFrame):
    m = SingleTableMetadata()
    m.detect_from_dataframe(df)
    return m


def _discretize(series: pd.Series, n_bins: int) -> np.ndarray:
    n_bins = min(n_bins, len(series) // 2)
    kbd = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
    return kbd.fit_transform(series.values.reshape(-1, 1)).ravel().astype(int)


# ── Optuna objectives ─────────────────────────────────────────────────────────
def _tune_gaussian_copula(df: pd.DataFrame, n_rows: int, n_trials: int):
    """GaussianCopula has few tuneable params; we tune the numerical distribution."""
    if not _SDV:
        return None, None, None
    meta = _sdv_meta(df)
    DISTS = ["norm", "beta", "truncnorm", "uniform", "gamma", "gaussian_kde"]

    def objective(trial):
        dist = trial.suggest_categorical("default_distribution", DISTS)
        s = GaussianCopulaSynthesizer(meta, default_distribution=dist)
        s.fit(df)
        return _quality(df, s.sample(num_rows=n_rows))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=min(n_trials, len(DISTS)), show_progress_bar=False)

    p = study.best_params
    s = GaussianCopulaSynthesizer(meta, default_distribution=p["default_distribution"])
    s.fit(df)
    return s.sample(num_rows=n_rows), study.best_value, p


def _tune_tvae(df: pd.DataFrame, n_rows: int, n_trials: int):
    if not _SDV:
        return None, None, None
    meta = _sdv_meta(df)

    def objective(trial):
        dim = trial.suggest_categorical("dim", [64, 128, 256])
        s = TVAESynthesizer(
            meta,
            epochs=trial.suggest_int("epochs", 100, 500, step=50),
            embedding_dim=trial.suggest_categorical("embedding_dim", [32, 64, 128]),
            compress_dims=(dim, dim),
            decompress_dims=(dim, dim),
            batch_size=trial.suggest_categorical("batch_size", [100, 250, 500]),
            l2scale=trial.suggest_float("l2scale", 1e-6, 1e-3, log=True),
        )
        s.fit(df)
        return _quality(df, s.sample(num_rows=n_rows))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    p = study.best_params
    dim = p["dim"]
    s = TVAESynthesizer(
        meta,
        epochs=p["epochs"], embedding_dim=p["embedding_dim"],
        compress_dims=(dim, dim), decompress_dims=(dim, dim),
        batch_size=p["batch_size"], l2scale=p["l2scale"],
    )
    s.fit(df)
    return s.sample(num_rows=n_rows), study.best_value, p


def _tune_ctgan(df: pd.DataFrame, n_rows: int, n_trials: int, cls=None):
    if not _SDV:
        return None, None, None
    if cls is None:
        cls = CTGANSynthesizer
    meta = _sdv_meta(df)

    def objective(trial):
        gdim = trial.suggest_categorical("gdim", [128, 256])
        ddim = trial.suggest_categorical("ddim", [128, 256])
        s = cls(
            meta,
            epochs=trial.suggest_int("epochs", 100, 500, step=50),
            embedding_dim=trial.suggest_categorical("embedding_dim", [32, 64, 128]),
            generator_dim=(gdim, gdim),
            discriminator_dim=(ddim, ddim),
            batch_size=trial.suggest_categorical("batch_size", [100, 250, 500]),
            discriminator_steps=trial.suggest_int("discriminator_steps", 1, 5),
        )
        s.fit(df)
        return _quality(df, s.sample(num_rows=n_rows))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    p = study.best_params
    s = cls(
        meta,
        epochs=p["epochs"], embedding_dim=p["embedding_dim"],
        generator_dim=(p["gdim"], p["gdim"]),
        discriminator_dim=(p["ddim"], p["ddim"]),
        batch_size=p["batch_size"],
        discriminator_steps=p["discriminator_steps"],
    )
    s.fit(df)
    return s.sample(num_rows=n_rows), study.best_value, p


def _tune_forest_diffusion(df: pd.DataFrame, n_rows: int, n_trials: int):
    if not _FOREST:
        return None, None, None
    X = df.values.astype(float)

    def objective(trial):
        model = ForestDiffusionModel(
            X,
            n_t=trial.suggest_int("n_t", 20, 100),
            duplicate_K=trial.suggest_categorical("duplicate_K", [50, 100, 200]),
            diffusion_type=trial.suggest_categorical("diffusion_type", ["flow", "vp"]),
            n_jobs=-1,
        )
        syn = pd.DataFrame(model.generate(batch_size=n_rows), columns=df.columns)
        return _quality(df, syn)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    p = study.best_params
    model = ForestDiffusionModel(
        X, n_t=p["n_t"], duplicate_K=p["duplicate_K"],
        diffusion_type=p["diffusion_type"], n_jobs=-1,
    )
    syn = pd.DataFrame(model.generate(batch_size=n_rows), columns=df.columns)
    return syn, study.best_value, p


def _tune_tabpfn(df: pd.DataFrame, target: str, n_rows: int, n_trials: int, api_key: str):
    if not _TABPFN:
        return None, None, None
    if _tabpfn_auth is not None:
        if not api_key:
            warnings.warn("TabPFN: no API key — skipping.  Pass --tabpfn-key YOUR_KEY")
            return None, None, None
        _tabpfn_auth(api_key)

    feature_cols = [c for c in df.columns if c != target]
    X = df[feature_cols].values.astype(float)
    y = df[target].values

    # TabPFN has minimal tuneable params; we tune the jitter scale
    def objective(trial):
        scale = trial.suggest_float("jitter_scale", 0.01, 0.15)
        model = _TabPFNReg()
        model.fit(X, y)
        idx   = np.random.randint(0, len(X), n_rows)
        sigma = X.std(axis=0).clip(min=1e-6) * scale
        X_new = X[idx] + np.random.normal(0.0, sigma, (n_rows, X.shape[1]))
        y_new = model.predict(X_new)
        df_new = pd.DataFrame(X_new, columns=feature_cols)
        df_new[target] = y_new
        return _quality(df, df_new[df.columns])

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    p = study.best_params
    model = _TabPFNReg()
    model.fit(X, y)
    X_new = np.column_stack([
        np.random.uniform(X[:, i].min(), X[:, i].max(), n_rows)
        for i in range(X.shape[1])
    ])
    y_new = model.predict(X_new)
    df_new = pd.DataFrame(X_new, columns=feature_cols)
    df_new[target] = y_new
    return df_new[df.columns], study.best_value, p


# ── visualization ─────────────────────────────────────────────────────────────
def _visualize(results: dict, df_orig: pd.DataFrame,
               x_col: str, y_col: str, t_col: str, out_path: Path):
    CMAP = mcolors.LinearSegmentedColormap.from_list(
        "dark_pale_blue", ["#08306b", "#c6dbef"]
    )
    all_sets = {"original": (df_orig, 0.0, {})} | results
    n = len(all_sets)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), facecolor="white")
    axes_flat = np.array(axes).reshape(-1)

    for ax, (label, (df, score, params)) in zip(axes_flat, all_sets.items()):
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        px = x_col if x_col in df.columns else (num_cols[0] if num_cols else None)
        py = y_col if y_col in df.columns else (num_cols[1] if len(num_cols) > 1 else None)
        pt = t_col if t_col in df.columns else None

        if px and py:
            c = df[pt] if pt else df[px] + df[py]
            ax.scatter(df[px], df[py], c=c, cmap=CMAP, s=4, linewidths=0, alpha=0.8)
            ax.set_xlim(df[px].min(), df[px].max())
            ax.set_ylim(df[py].min(), df[py].max())

        title = label.replace("_", " ")
        if score:
            title += f"\nKS={score:.3f}"
        ax.set_title(title, fontsize=24, pad=10)
        ax.axis("off")

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    plt.tight_layout(pad=0.4, h_pad=1.0, w_pad=0.4)
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="white", format="jpeg")
    plt.close()
    print(f"  JPG  -> {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    args    = _parse()
    csv_in  = args.csv_path
    target  = args.target
    n_trials = args.trials
    out_dir = args.out or csv_in.parent
    stem    = csv_in.stem
    tabpfn_key = args.tabpfn_key or os.environ.get("TABPFN_API_KEY", "")

    df = pd.read_csv(csv_in)
    print(f"Loaded  {len(df)} rows x {df.shape[1]} cols  from  {csv_in}")

    if target not in df.columns:
        sys.exit(f"ERROR: target '{target}' not found. Columns: {list(df.columns)}")

    n = len(df)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    x_col = "x" if "x" in df.columns else (num_cols[0] if num_cols else "")
    y_col = "y" if "y" in df.columns else (num_cols[1] if len(num_cols) > 1 else "")
    out_dir.mkdir(parents=True, exist_ok=True)

    # results dict: name -> (df, best_ks_score, best_params)
    results: dict[str, tuple] = {}

    models = [
        ("gaussian_copula",  lambda: _tune_gaussian_copula(df, n, n_trials)),
        ("tvae",             lambda: _tune_tvae(df, n, n_trials)),
        ("ctgan",            lambda: _tune_ctgan(df, n, n_trials, CTGANSynthesizer if _SDV else None)),
        ("copula_gan",       lambda: _tune_ctgan(df, n, n_trials, CopulaGANSynthesizer if _SDV else None)),
        ("forest_diffusion", lambda: _tune_forest_diffusion(df, n, n_trials)),
        ("tabpfn",           lambda: _tune_tabpfn(df, target, n, n_trials, tabpfn_key)),
    ]

    print(f"\nRunning {n_trials} Optuna trials per model...\n")
    for name, fn in models:
        print(f"  [{name}] ...", end=" ", flush=True)
        result = _safe(name, fn)
        if result is None or result[0] is None:
            print("skipped")
            continue
        df_best, score, params = result
        results[name] = (df_best, score, params)

        # save best CSV
        csv_out = out_dir / f"{stem}_{name}_best.csv"
        df_best.to_csv(csv_out, index=False)
        print(f"KS={score:.4f}  best params={params}")
        print(f"         CSV -> {csv_out}")

    # summary
    print("\n-- Best KS scores (lower = more similar to real) --")
    for name, (_, score, _) in results.items():
        print(f"  {name:25s}  {score:.4f}")

    # visualization JPG
    jpg_path = out_dir / f"{stem}_optuna_best.jpg"
    _visualize(results, df, x_col, y_col, target, jpg_path)


if __name__ == "__main__":
    main()
