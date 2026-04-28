"""Exploratory demand analysis helpers for cafeteria sales data.

Expected input schema for raw/prepared sales data:

    date: date-like value convertible to ``pl.Date``
    meal_category: string category, for example "Menu 1" or "Pizza - Oliva"
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
import polars as pl


SALES_DATE_COL = "date"
SALES_CATEGORY_COL = "meal_category"
SALES_NAME_COL = "name"
SALES_COUNT_COL = "sales"

SCHEDULE_DATE_COL = "date"
SCHEDULE_EVENT_ID_COL = "event_id"
SCHEDULE_TYPE_COL = "event_type"
SCHEDULE_REGISTERED_COL = "registered_students"

MEAL_PLAN_DATE_COL = "date"
MEAL_PLAN_CATEGORY_COL = "meal_category"
MEAL_PLAN_DISH_COL = "dish_name"
MEAL_PLAN_RECIPE_COL = "recipe_id"

WASTE_DATE_COL = "date"
WASTE_INGREDIENT_COL = "ingredient_name"
WASTE_UNIT_COL = "unit"
WASTE_QUANTITY_COL = "waste_quantity"

RECIPE_ID_COL = "recipe_id"
RECIPE_DISH_COL = "dish_name"
RECIPE_INGREDIENT_COL = "ingredient_name"
RECIPE_UNIT_COL = "unit"
RECIPE_AMOUNT_COL = "recipe_amount"

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


@dataclass(frozen=True)
class ExtendedDemandExploration:
    """Container for exploratory outputs across all prepared NDA data sources."""

    total_sales_analysis: TotalSalesAnalysis
    daily_schedule_load: pl.DataFrame | None = None
    schedule_sales_effect: pl.DataFrame | None = None
    category_schedule_effect: pl.DataFrame | None = None
    menu_popularity: pl.DataFrame | None = None
    daily_menu_mix: pl.DataFrame | None = None
    daily_waste: pl.DataFrame | None = None
    ingredient_unit_waste: pl.DataFrame | None = None
    waste_calendar_effect: pl.DataFrame | None = None
    recipe_ingredient_exposure: pl.DataFrame | None = None
    ingredient_waste_when_planned: pl.DataFrame | None = None


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


def validate_columns(df: pl.DataFrame, required_columns: set[str], table_name: str) -> None:
    """Raise a clear error if a prepared NDA dataframe is missing columns."""

    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required {table_name} columns: {sorted(missing_columns)}")


def normalize_sales_frame(sales: pl.DataFrame) -> pl.DataFrame:
    """Cast prepared sales data to stable types used by the analysis functions."""

    validate_sales_schema(sales)
    return sales.with_columns(
        pl.col(SALES_DATE_COL).cast(pl.Date),
        pl.col(SALES_CATEGORY_COL).cast(pl.Utf8),
        pl.col(SALES_NAME_COL).cast(pl.Utf8),
        pl.col(SALES_COUNT_COL).cast(pl.Int64),
    )


def normalize_schedule_frame(schedule: pl.DataFrame) -> pl.DataFrame:
    """Normalize prepared university schedule rows.

    Expected input schema:

        date: date-like value of the lecture/tutorial day
        event_id: stable lecture/tutorial identifier; unique per scheduled event is ideal
        event_type: string such as "lecture", "tutorial", "seminar", "exam", "event"
        registered_students: numeric count of registered students for that event

    Optional columns are ignored by this module but useful for later modeling:
    start_time, end_time, faculty, campus, room, course_id, degree_program.

    The analysis treats registered_students as a demand-pressure proxy, not as
    literal cafeteria attendance. If one course has multiple parallel tutorials,
    keep each tutorial as a separate row only if students are registered to that
    tutorial separately; otherwise de-duplicate before passing the table in.
    """

    validate_columns(
        schedule,
        {
            SCHEDULE_DATE_COL,
            SCHEDULE_EVENT_ID_COL,
            SCHEDULE_TYPE_COL,
            SCHEDULE_REGISTERED_COL,
        },
        "schedule",
    )
    return schedule.with_columns(
        pl.col(SCHEDULE_DATE_COL).cast(pl.Date),
        pl.col(SCHEDULE_EVENT_ID_COL).cast(pl.Utf8),
        pl.col(SCHEDULE_TYPE_COL).cast(pl.Utf8),
        pl.col(SCHEDULE_REGISTERED_COL).cast(pl.Float64),
    )


def normalize_meal_plan_frame(meal_plan: pl.DataFrame) -> pl.DataFrame:
    """Normalize prepared meal-plan rows.

    Expected input schema:

        date: date-like service day
        meal_category: category matching sales where possible, e.g. "Menu 1"
        dish_name: displayed dish name
        recipe_id: recipe identifier; nullable if no recipe match is available

    Recommended optional columns for later modeling:
    is_vegan, is_vegetarian, contains_meat, contains_fish, price, cuisine,
    production_complexity, planned_portions, sold_out_before_close.

    Keep one row per offered dish/category/day. If a category has multiple
    alternatives on the same day, keep multiple rows and distinguish dish_name
    and recipe_id.
    """

    validate_columns(
        meal_plan,
        {
            MEAL_PLAN_DATE_COL,
            MEAL_PLAN_CATEGORY_COL,
            MEAL_PLAN_DISH_COL,
            MEAL_PLAN_RECIPE_COL,
        },
        "meal_plan",
    )
    return meal_plan.with_columns(
        pl.col(MEAL_PLAN_DATE_COL).cast(pl.Date),
        pl.col(MEAL_PLAN_CATEGORY_COL).cast(pl.Utf8),
        pl.col(MEAL_PLAN_DISH_COL).cast(pl.Utf8),
        pl.col(MEAL_PLAN_RECIPE_COL).cast(pl.Utf8),
    )


def normalize_waste_frame(waste: pl.DataFrame) -> pl.DataFrame:
    """Normalize prepared ingredient waste rows.

    Expected input schema:

        date: date-like waste recording day
        ingredient_name: canonical ingredient label after your own cleaning
        unit: one of "liter", "kilogram", "portion", "piece"
        waste_quantity: numeric waste amount in the stated unit

    Important: this module never converts units. All waste summaries group by
    (ingredient_name, unit), so "tomato / kilogram" and "tomato / piece" remain
    separate measurement series.
    """

    validate_columns(
        waste,
        {
            WASTE_DATE_COL,
            WASTE_INGREDIENT_COL,
            WASTE_UNIT_COL,
            WASTE_QUANTITY_COL,
        },
        "waste",
    )
    return waste.with_columns(
        pl.col(WASTE_DATE_COL).cast(pl.Date),
        pl.col(WASTE_INGREDIENT_COL).cast(pl.Utf8),
        pl.col(WASTE_UNIT_COL).cast(pl.Utf8),
        pl.col(WASTE_QUANTITY_COL).cast(pl.Float64),
    )


def normalize_recipe_frame(recipes: pl.DataFrame) -> pl.DataFrame:
    """Normalize prepared recipe ingredient rows.

    Expected input schema:

        recipe_id: stable recipe identifier matching meal_plan.recipe_id
        dish_name: recipe/dish name
        ingredient_name: canonical ingredient label matching waste where possible
        unit: unit used in the recipe amount
        recipe_amount: numeric amount used in the recipe PDF

    Unknown portion counts are acceptable. recipe_amount is used only as a
    relative exposure signal within the same recipe unit, never as a per-portion
    amount and never converted across units.
    """

    validate_columns(
        recipes,
        {
            RECIPE_ID_COL,
            RECIPE_DISH_COL,
            RECIPE_INGREDIENT_COL,
            RECIPE_UNIT_COL,
            RECIPE_AMOUNT_COL,
        },
        "recipes",
    )
    return recipes.with_columns(
        pl.col(RECIPE_ID_COL).cast(pl.Utf8),
        pl.col(RECIPE_DISH_COL).cast(pl.Utf8),
        pl.col(RECIPE_INGREDIENT_COL).cast(pl.Utf8),
        pl.col(RECIPE_UNIT_COL).cast(pl.Utf8),
        pl.col(RECIPE_AMOUNT_COL).cast(pl.Float64),
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


def summarize_total_sales() -> list[pl.Expr]:
    """Return common demand summary statistics for stakeholder-facing charts."""

    return [
        pl.len().alias("service_days"),
        pl.mean("total_sales").alias("avg_sales"),
        pl.median("total_sales").alias("median_sales"),
        pl.col("total_sales").quantile(0.25).alias("q25_sales"),
        pl.col("total_sales").quantile(0.75).alias("q75_sales"),
        pl.std("total_sales").alias("std_sales"),
    ]


def build_daily_category_sales(sales_with_calendar: pl.DataFrame) -> pl.DataFrame:
    """Aggregate sales to one row per day and meal category before summaries."""

    return (
        sales_with_calendar.group_by(
            SALES_DATE_COL,
            SALES_CATEGORY_COL,
            "weekday",
            "weekday_number",
            "season",
            "semester",
            "semester_season",
            "is_lecture_day",
            "academic_bucket",
        )
        .agg(pl.sum(SALES_COUNT_COL).alias("category_sales"))
        .sort(SALES_DATE_COL, SALES_CATEGORY_COL)
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
    daily_category_sales = build_daily_category_sales(sales_with_calendar)

    weekday_effect = ordered_bar_data(
        daily_total_sales.group_by("weekday").agg(summarize_total_sales()),
        "weekday",
        WEEKDAY_ORDER,
    )
    season_effect = ordered_bar_data(
        daily_total_sales.group_by("season").agg(summarize_total_sales()),
        "season",
        SEASON_ORDER,
    )
    semester_effect = ordered_bar_data(
        daily_total_sales.group_by("semester").agg(summarize_total_sales()),
        "semester",
        [term["semester"] for term in semester_terms],
    )
    academic_effect = ordered_bar_data(
        daily_total_sales.group_by("academic_bucket").agg(summarize_total_sales()),
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
        .agg(summarize_total_sales())
    )
    category_effect = (
        daily_category_sales.group_by(SALES_CATEGORY_COL)
        .agg(
            pl.len().alias("service_days_offered"),
            pl.sum("category_sales").alias("total_sales"),
            pl.mean("category_sales").alias("avg_daily_sales_when_offered"),
            pl.median("category_sales").alias("median_daily_sales_when_offered"),
        )
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


def run_total_sales_analysis(
    sales: pl.DataFrame,
    *,
    show_plots: bool = True,
    **calendar_kwargs: Any,
) -> TotalSalesAnalysis:
    """Compute total-sales analysis and optionally display all plots."""

    analysis = compute_total_sales_analysis(sales, **calendar_kwargs)

    if show_plots:
        import matplotlib.pyplot as plt

        from .exploratory_plots import plot_total_sales_analysis

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


def build_daily_schedule_load(schedule: pl.DataFrame) -> pl.DataFrame:
    """Aggregate university schedule rows to a daily campus demand-pressure proxy."""

    normalized_schedule = normalize_schedule_frame(schedule)
    type_pivot = (
        normalized_schedule.group_by(SCHEDULE_DATE_COL, SCHEDULE_TYPE_COL)
        .agg(
            pl.sum(SCHEDULE_REGISTERED_COL).alias("registered_students_by_type"),
            pl.n_unique(SCHEDULE_EVENT_ID_COL).alias("event_count_by_type"),
        )
        .pivot(
            index=SCHEDULE_DATE_COL,
            on=SCHEDULE_TYPE_COL,
            values=["registered_students_by_type", "event_count_by_type"],
            aggregate_function="sum",
        )
    )
    totals = (
        normalized_schedule.group_by(SCHEDULE_DATE_COL)
        .agg(
            pl.sum(SCHEDULE_REGISTERED_COL).alias("registered_students_total"),
            pl.n_unique(SCHEDULE_EVENT_ID_COL).alias("scheduled_event_count"),
            pl.mean(SCHEDULE_REGISTERED_COL).alias("avg_registered_students_per_event"),
        )
        .sort(SCHEDULE_DATE_COL)
    )
    return totals.join(type_pivot, on=SCHEDULE_DATE_COL, how="left")


def add_schedule_load_bucket(df: pl.DataFrame) -> pl.DataFrame:
    """Bucket scheduled load using quantiles from positive-load days only."""

    positive_load = df.filter(pl.col("registered_students_total") > 0)
    if positive_load.is_empty():
        return df.with_columns(schedule_load_bucket=pl.lit("no_scheduled_load"))

    thresholds = positive_load.select(
        low_max=pl.col("registered_students_total").quantile(0.33),
        medium_max=pl.col("registered_students_total").quantile(0.66),
    ).row(0, named=True)
    low_max = thresholds["low_max"]
    medium_max = thresholds["medium_max"]

    return df.with_columns(
        schedule_load_bucket=pl.when(pl.col("registered_students_total") <= 0)
        .then(pl.lit("no_scheduled_load"))
        .when(pl.col("registered_students_total") <= low_max)
        .then(pl.lit("low"))
        .when(pl.col("registered_students_total") <= medium_max)
        .then(pl.lit("medium"))
        .otherwise(pl.lit("high")),
    )


def compute_schedule_sales_effect(
    daily_total_sales: pl.DataFrame,
    daily_schedule_load: pl.DataFrame,
) -> pl.DataFrame:
    """Join daily sales to schedule pressure and bin days into comparable load groups."""

    joined = (
        daily_total_sales.join(daily_schedule_load, on=SALES_DATE_COL, how="left")
        .with_columns(pl.col("registered_students_total").fill_null(0))
    )
    return add_schedule_load_bucket(joined)


def summarize_schedule_sales_effect(schedule_sales_effect: pl.DataFrame) -> pl.DataFrame:
    """Summarize average total demand by academic period and schedule-load bucket."""

    return (
        schedule_sales_effect.group_by("academic_bucket", "schedule_load_bucket")
        .agg(
            pl.len().alias("service_days"),
            pl.mean("total_sales").alias("avg_sales"),
            pl.median("total_sales").alias("median_sales"),
            pl.mean("registered_students_total").alias("avg_registered_students_total"),
        )
        .sort("academic_bucket", "schedule_load_bucket")
    )


def compute_category_schedule_effect(
    sales_with_calendar: pl.DataFrame,
    daily_schedule_load: pl.DataFrame,
) -> pl.DataFrame:
    """Estimate how meal-category sales change with daily schedule pressure."""

    daily_category_sales = build_daily_category_sales(sales_with_calendar)
    joined = (
        daily_category_sales.join(daily_schedule_load, on=SALES_DATE_COL, how="left")
        .with_columns(pl.col("registered_students_total").fill_null(0))
    )
    return (
        add_schedule_load_bucket(joined)
        .group_by(SALES_CATEGORY_COL, "schedule_load_bucket")
        .agg(
            pl.len().alias("service_days_offered"),
            pl.mean("category_sales").alias("avg_sales"),
            pl.median("category_sales").alias("median_sales"),
            pl.col("category_sales").quantile(0.25).alias("q25_sales"),
            pl.col("category_sales").quantile(0.75).alias("q75_sales"),
        )
        .sort(SALES_CATEGORY_COL, "schedule_load_bucket")
    )


def compute_menu_popularity(sales: pl.DataFrame, meal_plan: pl.DataFrame) -> pl.DataFrame:
    """Rank dishes by observed demand when offered.

    This assumes sales.meal_category and meal_plan.meal_category are compatible.
    If sales.name already equals the dish label, this join acts as a consistency
    check; if sales only has category-level counts, this estimates dish/category
    demand from the planned dish in that category.
    """

    normalized_sales = normalize_sales_frame(sales)
    normalized_meal_plan = normalize_meal_plan_frame(meal_plan)
    return (
        normalized_sales.join(
            normalized_meal_plan,
            left_on=[SALES_DATE_COL, SALES_CATEGORY_COL],
            right_on=[MEAL_PLAN_DATE_COL, MEAL_PLAN_CATEGORY_COL],
            how="inner",
        )
        .group_by(MEAL_PLAN_DISH_COL, SALES_CATEGORY_COL, MEAL_PLAN_RECIPE_COL)
        .agg(
            pl.len().alias("offered_days"),
            pl.sum(SALES_COUNT_COL).alias("total_sales"),
            pl.mean(SALES_COUNT_COL).alias("avg_sales_when_offered"),
            pl.median(SALES_COUNT_COL).alias("median_sales_when_offered"),
        )
        .sort("avg_sales_when_offered", descending=True)
    )


def build_daily_menu_mix(meal_plan: pl.DataFrame) -> pl.DataFrame:
    """Aggregate meal-plan rows into daily offer-count features."""

    normalized_meal_plan = normalize_meal_plan_frame(meal_plan)
    return (
        normalized_meal_plan.group_by(MEAL_PLAN_DATE_COL)
        .agg(
            pl.len().alias("offered_dish_count"),
            pl.n_unique(MEAL_PLAN_CATEGORY_COL).alias("offered_category_count"),
            pl.n_unique(MEAL_PLAN_RECIPE_COL).alias("offered_recipe_count"),
        )
        .sort(MEAL_PLAN_DATE_COL)
    )


def build_daily_waste(waste: pl.DataFrame) -> pl.DataFrame:
    """Aggregate waste without mixing measurement units."""

    normalized_waste = normalize_waste_frame(waste)
    return (
        normalized_waste.group_by(WASTE_DATE_COL, WASTE_INGREDIENT_COL, WASTE_UNIT_COL)
        .agg(pl.sum(WASTE_QUANTITY_COL).alias("daily_waste_quantity"))
        .sort(WASTE_DATE_COL, WASTE_INGREDIENT_COL, WASTE_UNIT_COL)
    )


def summarize_ingredient_unit_waste(daily_waste: pl.DataFrame) -> pl.DataFrame:
    """Rank waste series by ingredient and unit without unit conversion."""

    return (
        daily_waste.group_by(WASTE_INGREDIENT_COL, WASTE_UNIT_COL)
        .agg(
            pl.len().alias("waste_recorded_days"),
            pl.sum("daily_waste_quantity").alias("total_waste_quantity"),
            pl.mean("daily_waste_quantity").alias("avg_daily_waste_quantity"),
            pl.median("daily_waste_quantity").alias("median_daily_waste_quantity"),
            pl.std("daily_waste_quantity").alias("std_daily_waste_quantity"),
        )
        .sort(["unit", "total_waste_quantity"], descending=[False, True])
    )


def compute_waste_calendar_effect(daily_waste: pl.DataFrame, calendar: pl.DataFrame) -> pl.DataFrame:
    """Summarize ingredient-unit waste by weekday and academic period.

    Missing ingredient-unit waste records are treated as zero waste on regular
    service days in the observed waste date range.
    """

    if daily_waste.is_empty():
        return pl.DataFrame(
            schema={
                WASTE_INGREDIENT_COL: pl.Utf8,
                WASTE_UNIT_COL: pl.Utf8,
                "weekday": pl.Utf8,
                "academic_bucket": pl.Utf8,
                "service_days": pl.UInt32,
                "avg_waste_quantity": pl.Float64,
                "median_waste_quantity": pl.Float64,
                "q25_waste_quantity": pl.Float64,
                "q75_waste_quantity": pl.Float64,
            }
        )

    bounds = daily_waste.select(
        min_date=pl.min(WASTE_DATE_COL),
        max_date=pl.max(WASTE_DATE_COL),
    ).row(0, named=True)
    service_days = (
        calendar.filter(
            pl.col("is_regular_service_day")
            & pl.col("day").is_between(bounds["min_date"], bounds["max_date"], closed="both")
        )
        .select(
            pl.col("day").alias(WASTE_DATE_COL),
            "weekday",
            "academic_bucket",
        )
    )
    ingredient_units = daily_waste.select(WASTE_INGREDIENT_COL, WASTE_UNIT_COL).unique()
    full_waste_grid = (
        service_days.join(ingredient_units, how="cross")
        .join(
            daily_waste,
            on=[WASTE_DATE_COL, WASTE_INGREDIENT_COL, WASTE_UNIT_COL],
            how="left",
        )
        .with_columns(pl.col("daily_waste_quantity").fill_null(0))
    )

    return (
        full_waste_grid
        .group_by(WASTE_INGREDIENT_COL, WASTE_UNIT_COL, "weekday", "academic_bucket")
        .agg(
            pl.len().alias("service_days"),
            pl.mean("daily_waste_quantity").alias("avg_waste_quantity"),
            pl.median("daily_waste_quantity").alias("median_waste_quantity"),
            pl.col("daily_waste_quantity").quantile(0.25).alias("q25_waste_quantity"),
            pl.col("daily_waste_quantity").quantile(0.75).alias("q75_waste_quantity"),
        )
        .sort(WASTE_INGREDIENT_COL, WASTE_UNIT_COL, "academic_bucket", "weekday")
    )


def build_recipe_ingredient_exposure(meal_plan: pl.DataFrame, recipes: pl.DataFrame) -> pl.DataFrame:
    """Map planned menu days to recipe ingredient exposure.

    Because recipe portion counts are unknown, this table is not a production
    quantity estimate. It only says that an ingredient appeared on the menu and
    records the raw recipe amount as a relative signal within the same unit.
    """

    normalized_meal_plan = normalize_meal_plan_frame(meal_plan)
    normalized_recipes = normalize_recipe_frame(recipes)
    return (
        normalized_meal_plan.join(normalized_recipes, on=MEAL_PLAN_RECIPE_COL, how="inner")
        .group_by(MEAL_PLAN_DATE_COL, RECIPE_INGREDIENT_COL, RECIPE_UNIT_COL)
        .agg(
            pl.n_unique(MEAL_PLAN_RECIPE_COL).alias("planned_recipe_count"),
            pl.sum(RECIPE_AMOUNT_COL).alias("raw_recipe_amount_exposure"),
        )
        .sort(MEAL_PLAN_DATE_COL, RECIPE_INGREDIENT_COL, RECIPE_UNIT_COL)
    )


def compare_waste_when_ingredient_planned(
    daily_waste: pl.DataFrame,
    recipe_ingredient_exposure: pl.DataFrame,
) -> pl.DataFrame:
    """Compare waste on days where an ingredient-unit appears in planned recipes.

    This join only matches identical ingredient names and identical units. Rows
    with no matching recipe exposure are treated as "not planned in available
    recipes", not proof that the ingredient was unused.
    """

    if daily_waste.is_empty() and recipe_ingredient_exposure.is_empty():
        return pl.DataFrame(
            schema={
                WASTE_INGREDIENT_COL: pl.Utf8,
                WASTE_UNIT_COL: pl.Utf8,
                "ingredient_planned_in_available_recipes": pl.Boolean,
                "days": pl.UInt32,
                "avg_waste_quantity": pl.Float64,
                "median_waste_quantity": pl.Float64,
                "avg_planned_recipe_count": pl.Float64,
            }
        )

    exposure = recipe_ingredient_exposure
    observed_dates = pl.concat(
        [
            daily_waste.select(pl.col(WASTE_DATE_COL)),
            exposure.select(pl.col(WASTE_DATE_COL)),
        ],
        how="vertical",
    )
    bounds = observed_dates.select(
        min_date=pl.min(WASTE_DATE_COL),
        max_date=pl.max(WASTE_DATE_COL),
    ).row(0, named=True)
    dates = pl.DataFrame(
        {
            WASTE_DATE_COL: pl.date_range(
                bounds["min_date"],
                bounds["max_date"],
                interval="1d",
                eager=True,
            )
        }
    )
    ingredient_units = pl.concat(
        [
            daily_waste.select(WASTE_INGREDIENT_COL, WASTE_UNIT_COL),
            exposure.select(WASTE_INGREDIENT_COL, WASTE_UNIT_COL),
        ],
        how="vertical",
    ).unique()

    full_grid = (
        dates.join(ingredient_units, how="cross")
        .join(
            daily_waste,
            on=[WASTE_DATE_COL, WASTE_INGREDIENT_COL, WASTE_UNIT_COL],
            how="left",
        )
        .join(
            exposure,
            on=[WASTE_DATE_COL, WASTE_INGREDIENT_COL, WASTE_UNIT_COL],
            how="left",
        )
        .with_columns(
            ingredient_planned_in_available_recipes=pl.col("planned_recipe_count").is_not_null(),
            planned_recipe_count=pl.col("planned_recipe_count").fill_null(0),
            raw_recipe_amount_exposure=pl.col("raw_recipe_amount_exposure").fill_null(0),
            daily_waste_quantity=pl.col("daily_waste_quantity").fill_null(0),
        )
    )

    return (
        full_grid
        .group_by(WASTE_INGREDIENT_COL, WASTE_UNIT_COL, "ingredient_planned_in_available_recipes")
        .agg(
            pl.len().alias("days"),
            pl.mean("daily_waste_quantity").alias("avg_waste_quantity"),
            pl.median("daily_waste_quantity").alias("median_waste_quantity"),
            pl.col("daily_waste_quantity").quantile(0.25).alias("q25_waste_quantity"),
            pl.col("daily_waste_quantity").quantile(0.75).alias("q75_waste_quantity"),
            pl.mean("planned_recipe_count").alias("avg_planned_recipe_count"),
        )
        .sort(WASTE_INGREDIENT_COL, WASTE_UNIT_COL, "ingredient_planned_in_available_recipes")
    )


def compute_extended_demand_exploration(
    sales: pl.DataFrame,
    *,
    schedule: pl.DataFrame | None = None,
    meal_plan: pl.DataFrame | None = None,
    waste: pl.DataFrame | None = None,
    recipes: pl.DataFrame | None = None,
    **calendar_kwargs: Any,
) -> ExtendedDemandExploration:
    """Compute a broader EDA for demand prediction and waste reduction.

    Minimal required input is the existing sales table. Pass optional prepared
    NDA tables as they become available:

        schedule: see normalize_schedule_frame()
        meal_plan: see normalize_meal_plan_frame()
        waste: see normalize_waste_frame()
        recipes: see normalize_recipe_frame()

    Recommended analysis order:
    1. Total demand baseline: weekday, semester, lecture-period effects.
    2. Schedule pressure: does registered campus load explain sales residuals?
    3. Menu mix: how much does offer breadth vary by day?
    4. Waste concentration: which ingredient-unit series dominate waste?
    5. Recipe bridge: which ingredient-unit waste increases when planned?
    """

    total_sales_analysis = compute_total_sales_analysis(sales, **calendar_kwargs)

    daily_schedule_load = None
    schedule_sales_effect = None
    category_schedule_effect = None
    if schedule is not None:
        daily_schedule_load = build_daily_schedule_load(schedule)
        schedule_sales_effect = compute_schedule_sales_effect(
            total_sales_analysis.daily_total_sales,
            daily_schedule_load,
        )
        category_schedule_effect = compute_category_schedule_effect(
            total_sales_analysis.sales_with_calendar,
            daily_schedule_load,
        )

    menu_popularity = None
    daily_menu_mix = None
    if meal_plan is not None:
        daily_menu_mix = build_daily_menu_mix(meal_plan)

    daily_waste = None
    ingredient_unit_waste = None
    waste_calendar_effect = None
    if waste is not None:
        daily_waste = build_daily_waste(waste)
        ingredient_unit_waste = summarize_ingredient_unit_waste(daily_waste)
        waste_calendar_effect = compute_waste_calendar_effect(
            daily_waste,
            total_sales_analysis.calendar,
        )

    recipe_ingredient_exposure = None
    ingredient_waste_when_planned = None
    if meal_plan is not None and recipes is not None:
        recipe_ingredient_exposure = build_recipe_ingredient_exposure(meal_plan, recipes)
        if daily_waste is not None:
            ingredient_waste_when_planned = compare_waste_when_ingredient_planned(
                daily_waste,
                recipe_ingredient_exposure,
            )

    return ExtendedDemandExploration(
        total_sales_analysis=total_sales_analysis,
        daily_schedule_load=daily_schedule_load,
        schedule_sales_effect=schedule_sales_effect,
        category_schedule_effect=category_schedule_effect,
        menu_popularity=menu_popularity,
        daily_menu_mix=daily_menu_mix,
        daily_waste=daily_waste,
        ingredient_unit_waste=ingredient_unit_waste,
        waste_calendar_effect=waste_calendar_effect,
        recipe_ingredient_exposure=recipe_ingredient_exposure,
        ingredient_waste_when_planned=ingredient_waste_when_planned,
    )
