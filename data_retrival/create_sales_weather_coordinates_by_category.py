import argparse
import re
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def slugify(text: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9]+", "_", str(text).strip())
    clean = clean.strip("_").lower()
    return clean or "unknown_category"


def load_sales_by_category(sales_csv: Path) -> pd.DataFrame:
    sales_df = pd.read_csv(sales_csv)
    sales_df["date"] = pd.to_datetime(sales_df["date"], errors="coerce").dt.date
    sales_df = sales_df.dropna(subset=["date", "meal_category", "sales"])

    # Multiple rows can exist per category/day, so aggregate to one point per day.
    return (
        sales_df.groupby(["meal_category", "date"], as_index=False)["sales"]
        .sum()
        .sort_values(["meal_category", "date"])
    )


def load_weather(weather_csv: Path) -> pd.DataFrame:
    weather_df = pd.read_csv(weather_csv)
    weather_df["date"] = pd.to_datetime(
        weather_df["date"].astype(str).str.slice(0, 10), errors="coerce"
    ).dt.date
    needed_cols = ["date", "apparent_temperature", "precipitation", "cloud_cover"]
    return weather_df[needed_cols].dropna(subset=["date"])


def save_pair_csv(
    merged_df: pd.DataFrame,
    weather_col: str,
    out_path: Path,
) -> None:
    pair_df = merged_df[["date", "sales", weather_col]].rename(
        columns={"sales": "x_sales", weather_col: f"y_{weather_col}"}
    )
    pair_df.to_csv(out_path, index=False)


def save_pair_plot(
    merged_df: pd.DataFrame,
    weather_col: str,
    y_label: str,
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11.25, 6))
    ax.scatter(merged_df["sales"], merged_df[weather_col], alpha=0.7, s=28)
    ax.set_title(title)
    ax.set_xlabel("Sales")
    ax.set_ylabel(y_label)
    ax.grid(True, linestyle="--", alpha=0.35)
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", label="Se: WiSe 24/25", markerfacecolor=(0.9, 0.2, 0.2), markersize=8),
        Line2D([0], [0], marker="o", color="w", label="Se: SoSe 25", markerfacecolor=(0.1, 0.7, 0.2), markersize=8),
        Line2D([0], [0], marker="o", color="w", label="Se: WiSe 25/26", markerfacecolor=(0.1, 0.35, 1.0), markersize=8),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
        borderaxespad=0.0,
        title="Semester",
    )
    fig.tight_layout(rect=(0, 0, 0.82, 1))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create category-specific coordinate datasets and plots for sales vs weather."
    )
    parser.add_argument(
        "--sales-csv",
        type=Path,
        default=Path("data_retrival/data/sales_2025_processed.csv"),
    )
    parser.add_argument(
        "--weather-csv",
        type=Path,
        default=Path("data_retrival/data/weather_weekday_daily_since_2025.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data_retrival/data/category_coordinates"),
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    sales_by_cat = load_sales_by_category(args.sales_csv)
    weather_df = load_weather(args.weather_csv)

    created_count = 0
    for category in sales_by_cat["meal_category"].dropna().unique():
        cat_df = sales_by_cat[sales_by_cat["meal_category"] == category].copy()
        merged = pd.merge(cat_df, weather_df, on="date", how="inner").sort_values("date")
        if merged.empty:
            continue

        category_slug = slugify(category)
        base = f"{category_slug}_sales"

        csv_a = args.out_dir / f"{base}_apparent_temperature_coordinates.csv"
        csv_p = args.out_dir / f"{base}_precipitation_coordinates.csv"
        csv_c = args.out_dir / f"{base}_cloud_cover_coordinates.csv"

        png_a = args.out_dir / f"{base}_apparent_temperature_coordinates.png"
        png_p = args.out_dir / f"{base}_precipitation_coordinates.png"
        png_c = args.out_dir / f"{base}_cloud_cover_coordinates.png"

        save_pair_csv(merged, "apparent_temperature", csv_a)
        save_pair_csv(merged, "precipitation", csv_p)
        save_pair_csv(merged, "cloud_cover", csv_c)

        save_pair_plot(
            merged,
            weather_col="apparent_temperature",
            y_label="Apparent Temperature",
            title=f"{category}: Sales vs Apparent Temperature",
            out_path=png_a,
        )
        save_pair_plot(
            merged,
            weather_col="precipitation",
            y_label="Precipitation",
            title=f"{category}: Sales vs Precipitation",
            out_path=png_p,
        )
        save_pair_plot(
            merged,
            weather_col="cloud_cover",
            y_label="Cloud Cover",
            title=f"{category}: Sales vs Cloud Cover",
            out_path=png_c,
        )
        created_count += 1

    print(f"Created category outputs for {created_count} categories in: {args.out_dir}")


if __name__ == "__main__":
    main()
