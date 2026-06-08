import argparse
import re
from pathlib import Path

import pandas as pd


WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def slugify(text: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9]+", "_", str(text).strip())
    clean = clean.strip("_").lower()
    return clean or "unknown_category"


def load_sales_by_category(sales_csv: Path) -> pd.DataFrame:
    sales_df = pd.read_csv(sales_csv)
    sales_df["date"] = pd.to_datetime(sales_df["date"], errors="coerce")
    sales_df = sales_df.dropna(subset=["date", "meal_category", "sales"])
    sales_df["date_only"] = sales_df["date"].dt.date

    # One sales value per category and date.
    sales_by_cat = (
        sales_df.groupby(["meal_category", "date_only"], as_index=False)["sales"]
        .sum()
        .rename(columns={"date_only": "date"})
    )
    sales_by_cat["date"] = pd.to_datetime(sales_by_cat["date"], errors="coerce")
    return sales_by_cat


def load_weather(weather_csv: Path) -> pd.DataFrame:
    weather_df = pd.read_csv(weather_csv)
    weather_df["date"] = pd.to_datetime(
        weather_df["date"].astype(str).str.slice(0, 10), errors="coerce"
    )
    needed_cols = ["date", "apparent_temperature", "precipitation", "cloud_cover"]
    return weather_df[needed_cols].dropna(subset=["date"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create category-specific sales-weather coordinates split by weekdays "
            "(Monday to Friday), saved as separate CSV files."
        )
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
        default=Path("data_retrival/data/category_weekday_coordinates"),
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    sales_by_cat = load_sales_by_category(args.sales_csv)
    weather_df = load_weather(args.weather_csv)

    merged = pd.merge(sales_by_cat, weather_df, on="date", how="inner").sort_values("date")
    merged["weekday"] = merged["date"].dt.day_name()
    merged = merged[merged["weekday"].isin(WEEKDAYS)].copy()

    file_count = 0
    for category in merged["meal_category"].dropna().unique():
        category_df = merged[merged["meal_category"] == category].copy()
        if category_df.empty:
            continue

        category_slug = slugify(category)
        for weekday in WEEKDAYS:
            weekday_df = category_df[category_df["weekday"] == weekday].copy()
            if weekday_df.empty:
                continue

            out_df = weekday_df[
                [
                    "date",
                    "meal_category",
                    "weekday",
                    "sales",
                    "apparent_temperature",
                    "precipitation",
                    "cloud_cover",
                ]
            ].rename(
                columns={
                    "sales": "x_sales",
                    "apparent_temperature": "y_apparent_temperature",
                    "precipitation": "y_precipitation",
                    "cloud_cover": "y_cloud_cover",
                }
            )
            out_df["date"] = out_df["date"].dt.date

            out_name = f"{category_slug}_sales_{weekday.lower()}_coordinates.csv"
            out_df.to_csv(args.out_dir / out_name, index=False)
            file_count += 1

    print(f"Created {file_count} weekday-differentiated CSV files in: {args.out_dir}")


if __name__ == "__main__":
    main()
