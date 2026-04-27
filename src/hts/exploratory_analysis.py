"""Exploratory demand analysis helpers for cafeteria sales data.

Expected input schema for raw/prepared sales data:

    date: date-like value convertible to ``pl.Date``
    meal category: string category, for example "Menu 1" or "Pizza - Oliva"
    name: string meal/dish name
    sales: integer or numeric count of dishes passed over the counter and sold

The module assumes one row per sold meal offering and day. Missing meal/day rows
mean the meal was not present in the provided data, not necessarily zero demand.
For total-demand analysis, rows are aggregated to daily total sales before
calendar effects are computed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import holidays
import matplotlib.pyplot as plt
import polars as pl


SALES_DATE_COL = "date"
SALES_CATEGORY_COL = "meal category"
SALES_NAME_COL = "name"
SALES_COUNT_COL = "sales"

WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
SEASON_ORDER = ["winter", "spring", "summer", "autumn"]
ACADEMIC_BUCKET_ORDER = [
    "semester_start_period",
    "lecture",
    "semester_lecture_free",
    "semester_break",
]


DEFAULT_SEMESTER_TERMS = [
    {
        "semester": "winter_2024_2025",
        "semester_season": "winter",
        "semester_start": date(2024, 10, 1),
        "semester_end": date(2025, 3, 31),
        "lecture_start": date(2024, 10, 21),
        "lecture_end": date(2025, 2, 15),
    },
    {
        "semester": "summer_2025",
        "semester_season": "summer",
        "semester_start": date(2025, 4, 1),
        "semester_end": date(2025, 9, 30),
        "lecture_start": date(2025, 4, 14),
        "lecture_end": date(2025, 7, 19),
    },
    {
        "semester": "winter_2025_2026",
        "semester_season": "winter",
        "semester_start": date(2025, 10, 1),
        "semester_end": date(2026, 3, 31),
        "lecture_start": date(2025, 10, 13),
        "lecture_end": date(2026, 2, 14),
    },
]

DEFAULT_UNIVERSITY_HOLIDAYS = {
    "winter_2024_2025_christmas_break": (
        date(2024, 12, 23),
        date(2025, 1, 5),
    ),
    "winter_2025_2026_christmas_break": (
        date(2025, 12, 22),
        date(2026, 1, 7),
    ),
}

DEFAULT_CAMPUS_EVENTS = {
    'Tagung "Aufklaerung jenseits der Oeffentlichkeit"': (
        date(2025, 9, 4),
        date(2025, 9, 5),
    ),
}


@dataclass(frozen=True)
class TotalSalesAnalysis:
    """Container for reusable analysis outputs."""

    calendar: pl.DataFrame
    cafeteria_calendar: pl.DataFrame
    regular_service_calendar: pl.DataFrame
    sales_with_calendar: pl.DataFrame
    daily_total_sales: pl.DataFrame
    weekday_effect: pl.DataFrame
    season_effect: pl.DataFrame
    semester_effect: pl.DataFrame
    academic_effect: pl.DataFrame
    lecture_weekday_effect: pl.DataFrame
    category_effect: pl.DataFrame


def _period_label(
    periods: list[dict[str, Any]],
    label_col: str,
    start_col: str,
    end_col: str,
) -> pl.Expr:
    expr = pl.lit(None, dtype=pl.Utf8)
    for period in periods:
        expr = (
            pl.when(pl.col("day").is_between(period[start_col], period[end_col], closed="both"))
            .then(pl.lit(period[label_col]))
            .otherwise(expr)
        )
    return expr


def _in_period(periods: list[dict[str, Any]], start_col: str, end_col: str) -> pl.Expr:
    expr = pl.lit(False)
    for period in periods:
        expr = expr | pl.col("day").is_between(period[start_col], period[end_col], closed="both")
    return expr


def _in_relative_lecture_start_window(
    semester_terms: list[dict[str, Any]],
    start_offset_days: int,
    end_offset_days: int,
) -> pl.Expr:
    expr = pl.lit(False)
    for period in semester_terms:
        window_start = period["lecture_start"] + timedelta(days=start_offset_days)
        window_end = period["lecture_start"] + timedelta(days=end_offset_days)
        expr = expr | pl.col("day").is_between(window_start, window_end, closed="both")
    return expr


def _date_range_rows(
    ranges: dict[str, tuple[date, date]],
    description_col: str,
) -> list[dict[str, Any]]:
    return [
        {
            "day": start + timedelta(days=offset),
            description_col: description,
        }
        for description, (start, end) in ranges.items()
        for offset in range((end - start).days + 1)
    ]


def _date_range_frame(
    ranges: dict[str, tuple[date, date]],
    description_col: str,
) -> pl.DataFrame:
    return pl.DataFrame(
        _date_range_rows(ranges, description_col),
        schema={"day": pl.Date, description_col: pl.Utf8},
    )


def build_calendar(
    year: int = 2025,
    *,
    germany_subdivision: str | None = "RP",
    semester_terms: list[dict[str, Any]] | None = None,
    university_holidays: dict[str, tuple[date, date]] | None = None,
    campus_events: dict[str, tuple[date, date]] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Build full-year, cafeteria-weekday, and regular-service calendars."""

    semester_terms = semester_terms or DEFAULT_SEMESTER_TERMS
    university_holidays = university_holidays or DEFAULT_UNIVERSITY_HOLIDAYS
    campus_events = campus_events or DEFAULT_CAMPUS_EVENTS

    public_holidays = holidays.country_holidays(
        "DE",
        subdiv=germany_subdivision,
        years=[year],
    )
    public_holiday_df = pl.DataFrame(
        [
            {"day": holiday_day, "public_holiday_name": holiday_name}
            for holiday_day, holiday_name in sorted(public_holidays.items())
        ],
        schema={"day": pl.Date, "public_holiday_name": pl.Utf8},
    )
    university_holiday_df = _date_range_frame(
        university_holidays,
        "university_holiday_name",
    )
    campus_events_df = _date_range_frame(
        campus_events,
        "campus_event_description",
    )

    calendar = (
        pl.DataFrame(
            {
                "day": pl.date_range(
                    date(year, 1, 1),
                    date(year, 12, 31),
                    interval="1d",
                    eager=True,
                )
            }
        )
        .with_columns(
            weekday_number=pl.col("day").dt.weekday(),
            weekday=pl.col("day").dt.strftime("%A"),
            iso_week=pl.col("day").dt.week(),
            month=pl.col("day").dt.month(),
            quarter=pl.col("day").dt.quarter(),
            day_of_year=pl.col("day").dt.ordinal_day(),
        )
        .with_columns(
            is_weekend=pl.col("weekday_number").is_in([6, 7]),
            season=pl.when(pl.col("month").is_in([3, 4, 5]))
            .then(pl.lit("spring"))
            .when(pl.col("month").is_in([6, 7, 8]))
            .then(pl.lit("summer"))
            .when(pl.col("month").is_in([9, 10, 11]))
            .then(pl.lit("autumn"))
            .otherwise(pl.lit("winter")),
            semester=_period_label(semester_terms, "semester", "semester_start", "semester_end"),
            semester_season=_period_label(
                semester_terms,
                "semester_season",
                "semester_start",
                "semester_end",
            ),
            is_semester_day=_in_period(semester_terms, "semester_start", "semester_end"),
            is_lecture_day=_in_period(semester_terms, "lecture_start", "lecture_end"),
            is_orientation_week=_in_relative_lecture_start_window(semester_terms, -7, -1),
            is_first_lecture_week=_in_relative_lecture_start_window(semester_terms, 0, 6),
        )
        .with_columns(
            is_semester_start_period=pl.col("is_orientation_week")
            | pl.col("is_first_lecture_week"),
        )
        .join(public_holiday_df, on="day", how="left")
        .join(university_holiday_df, on="day", how="left")
        .join(campus_events_df, on="day", how="left")
        .with_columns(
            is_public_holiday=pl.col("public_holiday_name").is_not_null(),
            is_university_holiday=pl.col("university_holiday_name").is_not_null(),
            is_campus_event=pl.col("campus_event_description").is_not_null(),
        )
        .with_columns(
            is_university_closure_proxy=pl.col("is_public_holiday")
            | pl.col("is_university_holiday"),
        )
        .with_columns(
            academic_bucket=pl.when(pl.col("is_university_closure_proxy"))
            .then(pl.lit("university_closure"))
            .when(pl.col("is_semester_start_period"))
            .then(pl.lit("semester_start_period"))
            .when(pl.col("is_lecture_day"))
            .then(pl.lit("lecture"))
            .when(pl.col("is_semester_day"))
            .then(pl.lit("semester_lecture_free"))
            .otherwise(pl.lit("semester_break")),
            is_cafeteria_relevant_day=~pl.col("is_weekend"),
            is_regular_service_day=(~pl.col("is_weekend"))
            & (~pl.col("is_university_closure_proxy")),
        )
    )

    cafeteria_calendar = calendar.filter(pl.col("is_cafeteria_relevant_day"))
    regular_service_calendar = calendar.filter(pl.col("is_regular_service_day"))
    return calendar, cafeteria_calendar, regular_service_calendar


def validate_sales_schema(sales: pl.DataFrame) -> None:
    """Raise a clear error if the prepared sales dataframe is missing columns."""

    required_columns = {
        SALES_DATE_COL,
        SALES_CATEGORY_COL,
        SALES_NAME_COL,
        SALES_COUNT_COL,
    }
    missing_columns = required_columns - set(sales.columns)
    if missing_columns:
        raise ValueError(f"Missing required sales columns: {sorted(missing_columns)}")


def normalize_sales_frame(sales: pl.DataFrame) -> pl.DataFrame:
    """Cast prepared sales data to stable types used by the analysis functions."""

    validate_sales_schema(sales)
    return sales.with_columns(
        pl.col(SALES_DATE_COL).cast(pl.Date),
        pl.col(SALES_CATEGORY_COL).cast(pl.Utf8),
        pl.col(SALES_NAME_COL).cast(pl.Utf8),
        pl.col(SALES_COUNT_COL).cast(pl.Int64),
    )


def join_sales_to_calendar(sales: pl.DataFrame, calendar: pl.DataFrame) -> pl.DataFrame:
    """Join prepared sales rows to calendar features."""

    normalized_sales = normalize_sales_frame(sales)
    return normalized_sales.join(
        calendar,
        left_on=SALES_DATE_COL,
        right_on="day",
        how="left",
    )


def build_daily_total_sales(sales_with_calendar: pl.DataFrame) -> pl.DataFrame:
    """Aggregate meal rows to daily total demand with calendar features."""

    return (
        sales_with_calendar.group_by(
            SALES_DATE_COL,
            "weekday",
            "weekday_number",
            "season",
            "semester",
            "semester_season",
            "is_lecture_day",
            "academic_bucket",
        )
        .agg(pl.sum(SALES_COUNT_COL).alias("total_sales"))
        .sort(SALES_DATE_COL)
        .with_columns(
            pl.col("total_sales")
            .rolling_mean(window_size=10, min_samples=3)
            .alias("rolling_mean_10_service_days"),
        )
    )


def ordered_bar_data(df: pl.DataFrame, label_col: str, order: list[str]) -> pl.DataFrame:
    """Return a dataframe sorted by a fixed stakeholder-facing label order."""

    order_df = pl.DataFrame({label_col: order, "sort_order": list(range(len(order)))})
    return df.join(order_df, on=label_col, how="inner").sort("sort_order").drop("sort_order")


def compute_total_sales_analysis(
    sales: pl.DataFrame,
    *,
    year: int = 2025,
    germany_subdivision: str | None = "RP",
    semester_terms: list[dict[str, Any]] | None = None,
    university_holidays: dict[str, tuple[date, date]] | None = None,
    campus_events: dict[str, tuple[date, date]] | None = None,
) -> TotalSalesAnalysis:
    """Build calendar, join sales, and compute summary tables for plotting."""

    semester_terms = semester_terms or DEFAULT_SEMESTER_TERMS
    calendar, cafeteria_calendar, regular_service_calendar = build_calendar(
        year=year,
        germany_subdivision=germany_subdivision,
        semester_terms=semester_terms,
        university_holidays=university_holidays,
        campus_events=campus_events,
    )
    sales_with_calendar = join_sales_to_calendar(sales, calendar)
    daily_total_sales = build_daily_total_sales(sales_with_calendar)

    weekday_effect = ordered_bar_data(
        daily_total_sales.group_by("weekday").agg(pl.mean("total_sales").alias("avg_sales")),
        "weekday",
        WEEKDAY_ORDER,
    )
    season_effect = ordered_bar_data(
        daily_total_sales.group_by("season").agg(pl.mean("total_sales").alias("avg_sales")),
        "season",
        SEASON_ORDER,
    )
    semester_effect = ordered_bar_data(
        daily_total_sales.group_by("semester").agg(pl.mean("total_sales").alias("avg_sales")),
        "semester",
        [term["semester"] for term in semester_terms],
    )
    academic_effect = ordered_bar_data(
        daily_total_sales.group_by("academic_bucket").agg(pl.mean("total_sales").alias("avg_sales")),
        "academic_bucket",
        ACADEMIC_BUCKET_ORDER,
    )
    lecture_weekday_effect = (
        daily_total_sales.with_columns(
            period_type=pl.when(pl.col("is_lecture_day"))
            .then(pl.lit("Lecture period"))
            .otherwise(pl.lit("Lecture-free period"))
        )
        .group_by("weekday", "period_type")
        .agg(pl.mean("total_sales").alias("avg_sales"))
    )
    category_effect = (
        sales_with_calendar.group_by(SALES_CATEGORY_COL)
        .agg(pl.sum(SALES_COUNT_COL).alias("total_sales"))
        .with_columns((pl.col("total_sales") / pl.sum("total_sales") * 100).alias("sales_share_pct"))
        .sort("total_sales")
    )

    return TotalSalesAnalysis(
        calendar=calendar,
        cafeteria_calendar=cafeteria_calendar,
        regular_service_calendar=regular_service_calendar,
        sales_with_calendar=sales_with_calendar,
        daily_total_sales=daily_total_sales,
        weekday_effect=weekday_effect,
        season_effect=season_effect,
        semester_effect=semester_effect,
        academic_effect=academic_effect,
        lecture_weekday_effect=lecture_weekday_effect,
        category_effect=category_effect,
    )


def plot_total_demand_over_time(daily_total_sales: pl.DataFrame) -> tuple[plt.Figure, plt.Axes]:
    """Plot daily total demand and a 10 service-day rolling average."""

    rows = daily_total_sales.to_dicts()
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(
        [row[SALES_DATE_COL] for row in rows],
        [row["total_sales"] for row in rows],
        color="#9aa0a6",
        linewidth=1,
        alpha=0.45,
        label="Daily total",
    )
    ax.plot(
        [row[SALES_DATE_COL] for row in rows],
        [row["rolling_mean_10_service_days"] for row in rows],
        color="#1f77b4",
        linewidth=2.5,
        label="10 service-day average",
    )
    ax.set_title("Total Daily Demand Over Time")
    ax.set_xlabel("")
    ax.set_ylabel("Dishes sold")
    ax.legend(frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig, ax


def plot_calendar_effects(analysis: TotalSalesAnalysis) -> tuple[plt.Figure, list[plt.Axes]]:
    """Plot weekday, season, semester, and academic-period effects."""

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes_grid = plt.subplots(2, 2, figsize=(12, 8))
    axes = [axes_grid[0, 0], axes_grid[0, 1], axes_grid[1, 0], axes_grid[1, 1]]
    effect_specs = [
        (axes[0], analysis.weekday_effect, "weekday", "Average Demand by Weekday"),
        (axes[1], analysis.season_effect, "season", "Average Demand by Season"),
        (axes[2], analysis.semester_effect, "semester", "Average Demand by Semester"),
        (axes[3], analysis.academic_effect, "academic_bucket", "Average Demand by Academic Period"),
    ]

    for ax, data, label_col, title in effect_specs:
        rows = data.to_dicts()
        ax.bar(
            [row[label_col] for row in rows],
            [row["avg_sales"] for row in rows],
            color="#4c78a8",
        )
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("Average dishes sold")
        ax.tick_params(axis="x", rotation=30)
        ax.bar_label(ax.containers[0], fmt="%.0f", padding=3)

    fig.tight_layout()
    return fig, axes


def plot_weekday_lecture_vs_free(
    lecture_weekday_effect: pl.DataFrame,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot five weekday pairs comparing lecture and lecture-free periods."""

    lookup = {
        (row["weekday"], row["period_type"]): row["avg_sales"]
        for row in lecture_weekday_effect.to_dicts()
    }
    lecture_values = [
        lookup.get((weekday, "Lecture period"), 0)
        for weekday in WEEKDAY_ORDER
    ]
    lecture_free_values = [
        lookup.get((weekday, "Lecture-free period"), 0)
        for weekday in WEEKDAY_ORDER
    ]
    x_positions = list(range(len(WEEKDAY_ORDER)))
    bar_width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5))
    lecture_bars = ax.bar(
        [position - bar_width / 2 for position in x_positions],
        lecture_values,
        width=bar_width,
        label="Lecture period",
        color="#4c78a8",
    )
    lecture_free_bars = ax.bar(
        [position + bar_width / 2 for position in x_positions],
        lecture_free_values,
        width=bar_width,
        label="Lecture-free period",
        color="#f58518",
    )
    ax.set_title("Average Demand by Weekday: Lecture vs Lecture-Free Period")
    ax.set_xlabel("")
    ax.set_ylabel("Average dishes sold")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(WEEKDAY_ORDER, rotation=0)
    ax.legend(frameon=False)
    ax.bar_label(lecture_bars, fmt="%.0f", padding=3)
    ax.bar_label(lecture_free_bars, fmt="%.0f", padding=3)
    fig.tight_layout()
    return fig, ax


def plot_category_share(category_effect: pl.DataFrame) -> tuple[plt.Figure, plt.Axes]:
    """Plot category share of annual demand as a horizontal bar chart."""

    rows = category_effect.to_dicts()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        [row[SALES_CATEGORY_COL] for row in rows],
        [row["sales_share_pct"] for row in rows],
        color="#59a14f",
    )
    ax.set_title("Share of Annual Demand by Category")
    ax.set_xlabel("Share of dishes sold (%)")
    ax.set_ylabel("")
    ax.bar_label(ax.containers[0], fmt="%.1f%%", padding=3)
    fig.tight_layout()
    return fig, ax


def plot_total_sales_analysis(analysis: TotalSalesAnalysis) -> list[plt.Figure]:
    """Create all stakeholder-facing total-sales plots."""

    figures = [
        plot_total_demand_over_time(analysis.daily_total_sales)[0],
        plot_calendar_effects(analysis)[0],
        plot_weekday_lecture_vs_free(analysis.lecture_weekday_effect)[0],
        plot_category_share(analysis.category_effect)[0],
    ]
    return figures


def run_total_sales_analysis(
    sales: pl.DataFrame,
    *,
    show_plots: bool = True,
    **calendar_kwargs: Any,
) -> TotalSalesAnalysis:
    """Compute total-sales analysis and optionally display all plots."""

    analysis = compute_total_sales_analysis(sales, **calendar_kwargs)

    if show_plots:
        figures = plot_total_sales_analysis(analysis)

        filenames = [
            "total_demand_over_time.png",
            "calendar_effects.png",
            "weekday_lecture_vs_free.png",
            "category_share.png",
        ]

        for fig, name in zip(figures, filenames):
            fig.savefig('../../plots/' + name, dpi=300)
            plt.close(fig)

    return analysis