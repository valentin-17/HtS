"""
Re-visualizes all existing optuna best JPGs with fontsize 24 and clean padding.
Overwrites existing JPGs in-place.

Usage:
    python data_retrival/revisualize_optuna.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

SHAPES_DIR = Path(__file__).parent / "data" / "shapes"
CMAP = mcolors.LinearSegmentedColormap.from_list(
    "dark_pale_blue", ["#08306b", "#c6dbef"]
)

KNOWN_MODELS = [
    "gaussian_copula", "tvae", "ctgan",
    "copula_gan", "forest_diffusion", "tabpfn",
]


def _scatter(ax, df, x_col, y_col, t_col):
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    px  = x_col if x_col in df.columns else (num[0] if num else None)
    py  = y_col if y_col in df.columns else (num[1] if len(num) > 1 else None)
    pt  = t_col if t_col in df.columns else None
    if px and py:
        c = df[pt] if pt else df[px] + df[py]
        ax.scatter(df[px], df[py], c=c, cmap=CMAP, s=4, linewidths=0, alpha=0.8)
        ax.set_xlim(df[px].min(), df[px].max())
        ax.set_ylim(df[py].min(), df[py].max())


def revisualize(stem: str, orig_csv: Path, best_csvs: dict[str, Path]):
    orig_df = pd.read_csv(orig_csv)
    num     = orig_df.select_dtypes(include=[np.number]).columns.tolist()
    x_col   = "x" if "x" in orig_df.columns else (num[0] if num else "")
    y_col   = "y" if "y" in orig_df.columns else (num[1] if len(num) > 1 else "")
    t_col   = "t" if "t" in orig_df.columns else (num[2] if len(num) > 2 else "")

    datasets = {"original": orig_df}
    for name, path in best_csvs.items():
        try:
            datasets[name + " (best)"] = pd.read_csv(path)
        except Exception:
            pass

    n     = len(datasets)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.5 * ncols, 5.5 * nrows),
        facecolor="white",
    )
    axes_flat = np.array(axes).reshape(-1)

    for ax, (label, df) in zip(axes_flat, datasets.items()):
        _scatter(ax, df, x_col, y_col, t_col)
        ax.set_title(label.replace("_", " "), fontsize=24, pad=10)
        ax.axis("off")

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    plt.tight_layout(pad=1.5, h_pad=2.5, w_pad=1.5)

    out = SHAPES_DIR / f"{stem}_optuna_best.jpg"
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor="white", format="jpeg")
    plt.close()
    print(f"  saved  {out.name}  ({n} panels)")


def main():
    stems = [
        p.stem.replace("_optuna_best", "")
        for p in SHAPES_DIR.glob("*_optuna_best.jpg")
    ]

    for stem in sorted(stems):
        orig_csv = SHAPES_DIR / f"{stem}.csv"
        if not orig_csv.exists():
            print(f"  [skip] {stem} — original CSV not found")
            continue

        best_csvs = {}
        for model in KNOWN_MODELS:
            path = SHAPES_DIR / f"{stem}_{model}_best.csv"
            if path.exists():
                best_csvs[model] = path

        if not best_csvs:
            print(f"  [skip] {stem} — no best CSVs found")
            continue

        print(f"  {stem}  ({len(best_csvs)} models)")
        revisualize(stem, orig_csv, best_csvs)

    print("\nDone.")


if __name__ == "__main__":
    main()
