import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def load_and_prepare_sales(sales_csv: Path) -> pd.DataFrame:
    sales_df = pd.read_csv(sales_csv)
    sales_df["date"] = pd.to_datetime(sales_df["date"], errors="coerce").dt.date

    # Multiple sales rows can exist per day, so build one daily sales value.
    daily_sales = (
        sales_df.dropna(subset=["date"])
        .groupby("date", as_index=False)["sales"]
        .sum()
    )
    return daily_sales


def load_and_prepare_weather(weather_csv: Path) -> pd.DataFrame:
    weather_df = pd.read_csv(weather_csv)
    # Keep local calendar date from strings like "2025-01-01 00:00:00+01:00".
    weather_df["date"] = pd.to_datetime(
        weather_df["date"].astype(str).str.slice(0, 10), errors="coerce"
    ).dt.date

    needed_cols = ["date", "apparent_temperature", "precipitation", "cloud_cover"]
    return weather_df[needed_cols].dropna(subset=["date"])


def save_pair_csv(
    merged_df: pd.DataFrame, weather_col: str, out_dir: Path, out_name: str
) -> None:
    pair_df = merged_df[["date", "sales", weather_col]].rename(
        columns={"sales": "x_sales", weather_col: f"y_{weather_col}"}
    )
    pair_df.to_csv(out_dir / out_name, index=False)


def save_pair_plot(
    merged_df: pd.DataFrame,
    weather_col: str,
    out_dir: Path,
    out_name: str,
    y_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(11.25, 6))
    ax.scatter(merged_df["sales"], merged_df[weather_col], alpha=0.7, s=28)
    ax.set_title(f"Sales vs {y_label}")
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
    fig.savefig(out_dir / out_name, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create coordinate datasets (sales, weather_metric) for matching dates."
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
        default=Path("data_retrival/data"),
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    sales_df = load_and_prepare_sales(args.sales_csv)
    weather_df = load_and_prepare_weather(args.weather_csv)

    # Keep only dates that exist in both files.
    merged = pd.merge(sales_df, weather_df, on="date", how="inner").sort_values("date")

    save_pair_csv(
        merged,
        weather_col="apparent_temperature",
        out_dir=args.out_dir,
        out_name="sales_apparent_temperature_coordinates.csv",
    )
    save_pair_csv(
        merged,
        weather_col="precipitation",
        out_dir=args.out_dir,
        out_name="sales_precipitation_coordinates.csv",
    )
    save_pair_csv(
        merged,
        weather_col="cloud_cover",
        out_dir=args.out_dir,
        out_name="sales_cloud_cover_coordinates.csv",
    )
    save_pair_plot(
        merged,
        weather_col="apparent_temperature",
        out_dir=args.out_dir,
        out_name="sales_apparent_temperature_coordinates.png",
        y_label="Apparent Temperature",
    )
    save_pair_plot(
        merged,
        weather_col="precipitation",
        out_dir=args.out_dir,
        out_name="sales_precipitation_coordinates.png",
        y_label="Precipitation",
    )
    save_pair_plot(
        merged,
        weather_col="cloud_cover",
        out_dir=args.out_dir,
        out_name="sales_cloud_cover_coordinates.png",
        y_label="Cloud Cover",
    )

    print("Created:")
    print(args.out_dir / "sales_apparent_temperature_coordinates.csv")
    print(args.out_dir / "sales_precipitation_coordinates.csv")
    print(args.out_dir / "sales_cloud_cover_coordinates.csv")
    print(args.out_dir / "sales_apparent_temperature_coordinates.png")
    print(args.out_dir / "sales_precipitation_coordinates.png")
    print(args.out_dir / "sales_cloud_cover_coordinates.png")


if __name__ == "__main__":
    main()
