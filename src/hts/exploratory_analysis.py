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
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import holidays
import polars as pl


SALES_DATE_COL = "date"
SALES_CATEGORY_COL = "meal_category"
SALES_NAME_COL = "name"
SALES_COUNT_COL = "sales"

SCHEDULE_DATE_COL = "date"
SCHEDULE_EVENT_ID_COL = "event_id"
SCHEDULE_COURSE_ID_COL = "course_id"
SCHEDULE_PARALLEL_GROUP_COL = "parallel_group_id"
SCHEDULE_TYPE_COL = "event_type"
SCHEDULE_REGISTERED_COL = "registered_students"
SCHEDULE_TIME_WEIGHT_COL = "attendance_weight"
SCHEDULE_PRESSURE_COL = "registered_students_parallel_adjusted_time_weighted"
SCHEDULE_START_TIME_COL = "start_time"
SCHEDULE_END_TIME_COL = "end_time"

MEAL_PLAN_DATE_COL = "date"
MEAL_PLAN_LOCATION_COL = "location"
MEAL_PLAN_CATEGORY_COL = "meal_category"
MEAL_PLAN_DISH_COL = "dish_name"
MEAL_PLAN_RECIPE_COL = "recipe_id"

WASTE_DATE_COL = "date"
WASTE_LOCATION_COL = "location"
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


def _parse_time_minutes(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.hour * 60 + value.minute
    if isinstance(value, time):
        return value.hour * 60 + value.minute

    text = str(value).strip()
    if not text:
        return None
    for separator in (":", "."):
        if separator in text:
            hour_text, minute_text, *_ = text.split(separator)
            if hour_text.strip().isdigit() and minute_text.strip().isdigit():
                hour = int(hour_text.strip())
                minute = int(minute_text.strip()[:2])
                if 0 <= hour <= 23 and 0 <= minute <= 59:
                    return hour * 60 + minute
                return None
    if text.isdigit():
        hour = int(text)
        if 0 <= hour <= 23:
            return hour * 60
    return None


def _attendance_weight_from_minutes(start_minutes: int, end_minutes: int) -> float:
    relevant_start = 10 * 60
    relevant_end = 16 * 60
    if end_minutes <= start_minutes:
        raise ValueError("Schedule end time must be after start time.")

    overlaps_relevant_window = end_minutes > relevant_start and start_minutes < relevant_end
    return 1.0 if overlaps_relevant_window else 0.5


def normalize_schedule_frame(schedule: pl.DataFrame) -> pl.DataFrame:
    """Normalize prepared university schedule rows.

    Supported input schema:

        date: concrete appointment date after expanding weekly, biweekly,
            block-course, and single-appointment schedules
        event_id: stable appointment-row identifier; unique per concrete
            scheduled appointment
        course_id: stable course identifier shared by all appointments that
            belong to the same course/module
        event_type: string such as "lecture", "tutorial", "seminar", "exam", "event"
        registered_students: numeric count of registered students for the
            course/appointment as reported in the source system
        start_time, end_time: appointment start/end times. ``attendance_weight``
            is always derived from the 10:00-16:00 cafeteria-relevant window:
            1.0 for any overlap, otherwise 0.5.

    Optional parallel-group column:

        parallel_group_id: identifier for known mutually exclusive parallel
            appointments where students only attend one option. If this column
            is missing or null, every concrete appointment is treated as its own
            appointment and no parallel relationship is inferred.

    Recommended preparation before passing the table:

        1. Expand all schedule patterns to one row per concrete appointment date.
        2. Do not infer cross-day parallel alternatives from weak signals such
           as similar registration counts or similar names.
        3. Leave parallel_group_id missing/null unless the raw data explicitly
           identifies a true alternative group.
        4. Only deduplicate obvious exact export duplicates before this step,
           for example identical date, course_id, event_type, start_time,
           end_time, room, and registered_students.

    Optional columns are ignored by this module but useful for later modeling:
    faculty, campus, room, degree_program, schedule_type.

    The analysis treats all registration-derived columns as demand-pressure
    proxies, not as literal cafeteria attendance.
    """

    required_columns = {
        SCHEDULE_DATE_COL,
        SCHEDULE_EVENT_ID_COL,
        SCHEDULE_COURSE_ID_COL,
        SCHEDULE_TYPE_COL,
        SCHEDULE_REGISTERED_COL,
        SCHEDULE_START_TIME_COL,
        SCHEDULE_END_TIME_COL,
    }
    validate_columns(
        schedule,
        required_columns,
        "schedule",
    )
    if SCHEDULE_PARALLEL_GROUP_COL in schedule.columns:
        schedule_prepared = schedule
    else:
        schedule_prepared = schedule.with_columns(
            pl.lit(None, dtype=pl.Utf8).alias(SCHEDULE_PARALLEL_GROUP_COL)
        )

    schedule_with_row_id = schedule_prepared.with_row_index("__schedule_row")
    weight_rows = []
    for row in schedule_with_row_id.select(
        "__schedule_row",
        SCHEDULE_EVENT_ID_COL,
        SCHEDULE_START_TIME_COL,
        SCHEDULE_END_TIME_COL,
    ).to_dicts():
        start_minutes = _parse_time_minutes(row[SCHEDULE_START_TIME_COL])
        end_minutes = _parse_time_minutes(row[SCHEDULE_END_TIME_COL])
        if start_minutes is None or end_minutes is None:
            raise ValueError(
                "Could not derive attendance_weight from start_time/end_time "
                f"for event_id={row[SCHEDULE_EVENT_ID_COL]!r}."
            )
        weight_rows.append(
            {
                "__schedule_row": row["__schedule_row"],
                SCHEDULE_TIME_WEIGHT_COL: _attendance_weight_from_minutes(
                    start_minutes,
                    end_minutes,
                ),
            }
        )
    normalized = (
        schedule_with_row_id.drop(SCHEDULE_TIME_WEIGHT_COL, strict=False)
        .join(
            pl.DataFrame(
                weight_rows,
                schema={
                    "__schedule_row": pl.UInt32,
                    SCHEDULE_TIME_WEIGHT_COL: pl.Float64,
                },
            ),
            on="__schedule_row",
            how="left",
        )
        .with_columns(pl.col(SCHEDULE_EVENT_ID_COL).cast(pl.Utf8))
        .with_columns(
            pl.col(SCHEDULE_DATE_COL).cast(pl.Date),
            pl.col(SCHEDULE_EVENT_ID_COL).cast(pl.Utf8),
            pl.col(SCHEDULE_COURSE_ID_COL).cast(pl.Utf8),
            pl.col(SCHEDULE_PARALLEL_GROUP_COL).cast(pl.Utf8).str.strip_chars(),
            pl.col(SCHEDULE_TYPE_COL).cast(pl.Utf8),
            pl.col(SCHEDULE_REGISTERED_COL).cast(pl.Float64).fill_null(0),
            pl.col(SCHEDULE_TIME_WEIGHT_COL).cast(pl.Float64).fill_null(0),
        )
        .with_columns(
            pl.col(SCHEDULE_TIME_WEIGHT_COL).clip(0, 1),
            parallel_group_id_normalized=pl.when(
                pl.col(SCHEDULE_PARALLEL_GROUP_COL).is_null()
                | (pl.col(SCHEDULE_PARALLEL_GROUP_COL) == "")
            )
            .then(pl.lit(None, dtype=pl.Utf8))
            .otherwise(pl.col(SCHEDULE_PARALLEL_GROUP_COL)),
        )
        .with_columns(
            effective_parallel_group_id=pl.when(pl.col("parallel_group_id_normalized").is_null())
            .then(pl.col(SCHEDULE_EVENT_ID_COL))
            .otherwise(pl.col("parallel_group_id_normalized")),
        )
        .with_columns(
            registered_students_time_weighted=(
                pl.col(SCHEDULE_REGISTERED_COL) * pl.col(SCHEDULE_TIME_WEIGHT_COL)
            ),
        )
        .drop("__schedule_row")
    )
    negative_rows = normalized.filter(pl.col(SCHEDULE_REGISTERED_COL) < 0)
    if not negative_rows.is_empty():
        examples = negative_rows.select(
            SCHEDULE_EVENT_ID_COL,
            SCHEDULE_COURSE_ID_COL,
            SCHEDULE_REGISTERED_COL,
        ).head(5)
        raise ValueError(
            "Schedule contains negative registered_students values: "
            f"{examples.to_dicts()}"
        )
    return normalized


def normalize_meal_plan_frame(meal_plan: pl.DataFrame) -> pl.DataFrame:
    """Normalize prepared meal-plan rows.

    Expected input schema:

        date: date-like service day
        location: tarforst or oliva
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
            MEAL_PLAN_LOCATION_COL,
            MEAL_PLAN_CATEGORY_COL,
            MEAL_PLAN_DISH_COL,
            MEAL_PLAN_RECIPE_COL,
        },
        "meal_plan",
    )
    return meal_plan.with_columns(
        pl.col(MEAL_PLAN_DATE_COL).cast(pl.Date),
        pl.col(MEAL_PLAN_LOCATION_COL).cast(pl.Utf8),
        pl.col(MEAL_PLAN_CATEGORY_COL).cast(pl.Utf8),
        pl.col(MEAL_PLAN_DISH_COL).cast(pl.Utf8),
        pl.col(MEAL_PLAN_RECIPE_COL).cast(pl.Utf8),
    )


def normalize_waste_frame(waste: pl.DataFrame) -> pl.DataFrame:
    """Normalize prepared ingredient waste rows.

    Expected input schema:

        date: date-like waste recording day
        location: waste processing location, for example "Mensa Tarforst"
        ingredient_name: canonical ingredient label after your own cleaning
        unit: one of "liter", "kilogram", "portion", "piece"
        waste_quantity: numeric waste amount in the stated unit

    Important: this module never converts units. All waste summaries group by
    (location, ingredient_name, unit), so locations and measurement series stay
    separate.
    """

    validate_columns(
        waste,
        {
            WASTE_DATE_COL,
            WASTE_LOCATION_COL,
            WASTE_INGREDIENT_COL,
            WASTE_UNIT_COL,
            WASTE_QUANTITY_COL,
        },
        "waste",
    )
    return waste.with_columns(
        pl.col(WASTE_DATE_COL).cast(pl.Date),
        pl.col(WASTE_LOCATION_COL).cast(pl.Utf8),
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
    schedule: pl.DataFrame | None = None,
    meal_plan: pl.DataFrame | None = None,
    waste: pl.DataFrame | None = None,
    recipes: pl.DataFrame | None = None,
    show_plots: bool = True,
    output_dir: str | Path = "../../plots",
    dpi: int = 300,
    **calendar_kwargs: Any,
) -> ExtendedDemandExploration:
    """Run the full exploratory analysis and save every available plot.

    Required input:

        sales: see normalize_sales_frame()

    Optional prepared NDA tables:

        schedule: see normalize_schedule_frame()
        meal_plan: see normalize_meal_plan_frame()
        waste: see normalize_waste_frame()
        recipes: see normalize_recipe_frame()

    The function is intentionally tolerant of missing optional tables. With only
    sales available, it computes and saves the baseline total-sales plots. As
    additional tables are supplied, it adds schedule, menu, waste, and recipe
    bridge plots to the same output directory.

    The historical ``show_plots`` parameter now controls whether plots are
    written to disk. It is kept to avoid breaking existing notebooks.
    """

    exploration = compute_extended_demand_exploration(
        sales,
        schedule=schedule,
        meal_plan=meal_plan,
        waste=waste,
        recipes=recipes,
        **calendar_kwargs,
    )

    if show_plots:
        import matplotlib.pyplot as plt

        from .exploratory_plots import named_extended_demand_exploration_plots

        plot_directory = Path(output_dir)
        plot_directory.mkdir(parents=True, exist_ok=True)

        for name, fig in named_extended_demand_exploration_plots(exploration):
            fig.savefig(plot_directory / name, dpi=dpi)
            plt.close(fig)

    return exploration


def build_daily_schedule_load(schedule: pl.DataFrame) -> pl.DataFrame:
    """Aggregate schedule rows to daily campus demand-pressure proxies.

    Main output columns:

        registered_students_raw: sum across all appointment rows; upper bound
        registered_students_course_deduplicated: one registration count per
            course/day, using max registered_students
        registered_students_parallel_adjusted: one registration count per
            course/day/parallel group, using max registered_students
        registered_students_time_weighted: raw appointment sum multiplied by
            attendance_weight
        registered_students_parallel_adjusted_time_weighted: default pressure
            proxy used for bucketed analysis and plots
    """

    normalized_schedule = normalize_schedule_frame(schedule)
    parallel_adjusted_rows = (
        normalized_schedule.group_by(
            SCHEDULE_DATE_COL,
            SCHEDULE_COURSE_ID_COL,
            "effective_parallel_group_id",
            SCHEDULE_TYPE_COL,
        )
        .agg(
            pl.max(SCHEDULE_REGISTERED_COL).alias("parallel_adjusted_registered_students"),
            pl.max("registered_students_time_weighted").alias(
                "parallel_adjusted_registered_students_time_weighted"
            ),
            pl.n_unique(SCHEDULE_EVENT_ID_COL).alias("appointment_rows_in_parallel_group"),
            pl.mean(SCHEDULE_TIME_WEIGHT_COL).alias("avg_parallel_group_time_weight"),
        )
    )
    course_deduplicated_rows = (
        normalized_schedule.group_by(SCHEDULE_DATE_COL, SCHEDULE_COURSE_ID_COL)
        .agg(
            pl.max(SCHEDULE_REGISTERED_COL).alias("course_deduplicated_registered_students"),
            pl.max("registered_students_time_weighted").alias(
                "course_deduplicated_registered_students_time_weighted"
            ),
        )
    )
    type_pivot = (
        parallel_adjusted_rows.group_by(SCHEDULE_DATE_COL, SCHEDULE_TYPE_COL)
        .agg(
            pl.sum("parallel_adjusted_registered_students").alias(
                "parallel_adjusted_registered_students_by_type"
            ),
            pl.sum("parallel_adjusted_registered_students_time_weighted").alias(
                "parallel_adjusted_time_weighted_registered_students_by_type"
            ),
            pl.len().alias("parallel_group_count_by_type"),
        )
        .pivot(
            index=SCHEDULE_DATE_COL,
            on=SCHEDULE_TYPE_COL,
            values=[
                "parallel_adjusted_registered_students_by_type",
                "parallel_adjusted_time_weighted_registered_students_by_type",
                "parallel_group_count_by_type",
            ],
            aggregate_function="sum",
        )
    )
    appointment_totals = (
        normalized_schedule.group_by(SCHEDULE_DATE_COL)
        .agg(
            pl.sum(SCHEDULE_REGISTERED_COL).alias("registered_students_raw"),
            pl.sum("registered_students_time_weighted").alias("registered_students_time_weighted"),
            pl.n_unique(SCHEDULE_EVENT_ID_COL).alias("scheduled_event_count"),
            pl.n_unique(SCHEDULE_COURSE_ID_COL).alias("scheduled_course_count"),
            pl.n_unique("effective_parallel_group_id").alias("parallel_group_count"),
            pl.mean(SCHEDULE_REGISTERED_COL).alias("avg_registered_students_per_event"),
            pl.mean(SCHEDULE_TIME_WEIGHT_COL).alias("avg_attendance_weight"),
        )
    )
    course_totals = (
        course_deduplicated_rows.group_by(SCHEDULE_DATE_COL)
        .agg(
            pl.sum("course_deduplicated_registered_students").alias(
                "registered_students_course_deduplicated"
            ),
            pl.sum("course_deduplicated_registered_students_time_weighted").alias(
                "registered_students_course_deduplicated_time_weighted"
            ),
        )
    )
    parallel_totals = (
        parallel_adjusted_rows.group_by(SCHEDULE_DATE_COL)
        .agg(
            pl.sum("parallel_adjusted_registered_students").alias(
                "registered_students_parallel_adjusted"
            ),
            pl.sum("parallel_adjusted_registered_students_time_weighted").alias(
                SCHEDULE_PRESSURE_COL
            ),
            pl.sum("appointment_rows_in_parallel_group").alias("scheduled_event_count_after_grouping"),
            pl.mean("avg_parallel_group_time_weight").alias("avg_parallel_adjusted_time_weight"),
        )
    )
    return (
        appointment_totals.join(course_totals, on=SCHEDULE_DATE_COL, how="left")
        .join(parallel_totals, on=SCHEDULE_DATE_COL, how="left")
        .join(type_pivot, on=SCHEDULE_DATE_COL, how="left")
        .sort(SCHEDULE_DATE_COL)
    )


def add_schedule_load_bucket(df: pl.DataFrame) -> pl.DataFrame:
    """Bucket scheduled load using quantiles from positive-load days only."""

    positive_load = df.filter(pl.col(SCHEDULE_PRESSURE_COL) > 0)
    if positive_load.is_empty():
        return df.with_columns(schedule_load_bucket=pl.lit("no_scheduled_load"))

    thresholds = positive_load.select(
        low_max=pl.col(SCHEDULE_PRESSURE_COL).quantile(0.33),
        medium_max=pl.col(SCHEDULE_PRESSURE_COL).quantile(0.66),
    ).row(0, named=True)
    low_max = thresholds["low_max"]
    medium_max = thresholds["medium_max"]

    return df.with_columns(
        schedule_load_bucket=pl.when(pl.col(SCHEDULE_PRESSURE_COL) <= 0)
        .then(pl.lit("no_scheduled_load"))
        .when(pl.col(SCHEDULE_PRESSURE_COL) <= low_max)
        .then(pl.lit("low"))
        .when(pl.col(SCHEDULE_PRESSURE_COL) <= medium_max)
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
        .with_columns(
            pl.col("registered_students_raw").fill_null(0),
            pl.col("registered_students_course_deduplicated").fill_null(0),
            pl.col("registered_students_parallel_adjusted").fill_null(0),
            pl.col("registered_students_time_weighted").fill_null(0),
            pl.col(SCHEDULE_PRESSURE_COL).fill_null(0),
        )
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
            pl.mean(SCHEDULE_PRESSURE_COL).alias("avg_schedule_pressure"),
            pl.mean("registered_students_raw").alias("avg_registered_students_raw"),
            pl.mean("registered_students_parallel_adjusted").alias(
                "avg_registered_students_parallel_adjusted"
            ),
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
        .with_columns(
            pl.col("registered_students_raw").fill_null(0),
            pl.col("registered_students_course_deduplicated").fill_null(0),
            pl.col("registered_students_parallel_adjusted").fill_null(0),
            pl.col("registered_students_time_weighted").fill_null(0),
            pl.col(SCHEDULE_PRESSURE_COL).fill_null(0),
        )
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
        normalized_waste.group_by(
            WASTE_DATE_COL,
            WASTE_LOCATION_COL,
            WASTE_INGREDIENT_COL,
            WASTE_UNIT_COL,
        )
        .agg(pl.sum(WASTE_QUANTITY_COL).alias("daily_waste_quantity"))
        .sort(WASTE_DATE_COL, WASTE_LOCATION_COL, WASTE_INGREDIENT_COL, WASTE_UNIT_COL)
    )


def summarize_ingredient_unit_waste(daily_waste: pl.DataFrame) -> pl.DataFrame:
    """Rank waste series by location, ingredient, and unit without unit conversion."""

    return (
        daily_waste.group_by(WASTE_LOCATION_COL, WASTE_INGREDIENT_COL, WASTE_UNIT_COL)
        .agg(
            pl.len().alias("waste_recorded_days"),
            pl.sum("daily_waste_quantity").alias("total_waste_quantity"),
            pl.mean("daily_waste_quantity").alias("avg_daily_waste_quantity"),
            pl.median("daily_waste_quantity").alias("median_daily_waste_quantity"),
            pl.std("daily_waste_quantity").alias("std_daily_waste_quantity"),
        )
        .sort([WASTE_LOCATION_COL, WASTE_UNIT_COL, "total_waste_quantity"], descending=[False, False, True])
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
                WASTE_LOCATION_COL: pl.Utf8,
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

    location_bounds = (
        daily_waste.group_by(WASTE_LOCATION_COL)
        .agg(
            pl.min(WASTE_DATE_COL).alias("min_date"),
            pl.max(WASTE_DATE_COL).alias("max_date"),
        )
    )
    location_service_days = (
        calendar.filter(pl.col("is_regular_service_day"))
        .select(
            pl.col("day").alias(WASTE_DATE_COL),
            "weekday",
            "academic_bucket",
        )
        .join(location_bounds, how="cross")
        .filter(pl.col(WASTE_DATE_COL).is_between(pl.col("min_date"), pl.col("max_date"), closed="both"))
        .drop("min_date", "max_date")
    )
    location_ingredient_units = daily_waste.select(
        WASTE_LOCATION_COL,
        WASTE_INGREDIENT_COL,
        WASTE_UNIT_COL,
    ).unique()
    full_waste_grid = (
        location_service_days.join(location_ingredient_units, on=WASTE_LOCATION_COL, how="inner")
        .join(
            daily_waste,
            on=[WASTE_DATE_COL, WASTE_LOCATION_COL, WASTE_INGREDIENT_COL, WASTE_UNIT_COL],
            how="left",
        )
        .with_columns(pl.col("daily_waste_quantity").fill_null(0))
    )

    return (
        full_waste_grid
        .group_by(WASTE_LOCATION_COL, WASTE_INGREDIENT_COL, WASTE_UNIT_COL, "weekday", "academic_bucket")
        .agg(
            pl.len().alias("service_days"),
            pl.mean("daily_waste_quantity").alias("avg_waste_quantity"),
            pl.median("daily_waste_quantity").alias("median_waste_quantity"),
            pl.col("daily_waste_quantity").quantile(0.25).alias("q25_waste_quantity"),
            pl.col("daily_waste_quantity").quantile(0.75).alias("q75_waste_quantity"),
        )
        .sort(WASTE_LOCATION_COL, WASTE_INGREDIENT_COL, WASTE_UNIT_COL, "academic_bucket", "weekday")
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
        .group_by(MEAL_PLAN_DATE_COL, MEAL_PLAN_LOCATION_COL, RECIPE_INGREDIENT_COL, RECIPE_UNIT_COL)
        .agg(
            pl.n_unique(MEAL_PLAN_RECIPE_COL).alias("planned_recipe_count"),
            pl.sum(RECIPE_AMOUNT_COL).alias("raw_recipe_amount_exposure"),
        )
        .sort(MEAL_PLAN_DATE_COL, MEAL_PLAN_LOCATION_COL, RECIPE_INGREDIENT_COL, RECIPE_UNIT_COL)
    )


def compare_waste_when_ingredient_planned(
    daily_waste: pl.DataFrame,
    recipe_ingredient_exposure: pl.DataFrame,
    calendar: pl.DataFrame,
) -> pl.DataFrame:
    """Compare waste on days where an ingredient-unit appears in planned recipes.

    This join only matches identical locations, ingredient names, and units.
    Rows with no matching recipe exposure are treated as "not planned in
    available recipes", not proof that the ingredient was unused.
    """

    if daily_waste.is_empty() and recipe_ingredient_exposure.is_empty():
        return pl.DataFrame(
            schema={
                WASTE_INGREDIENT_COL: pl.Utf8,
                WASTE_LOCATION_COL: pl.Utf8,
                WASTE_UNIT_COL: pl.Utf8,
                "ingredient_planned_in_available_recipes": pl.Boolean,
                "days": pl.UInt32,
                "avg_waste_quantity": pl.Float64,
                "median_waste_quantity": pl.Float64,
                "q25_waste_quantity": pl.Float64,
                "q75_waste_quantity": pl.Float64,
                "avg_planned_recipe_count": pl.Float64,
            }
        )

    exposure = recipe_ingredient_exposure
    location_bounds = (
        daily_waste.group_by(WASTE_LOCATION_COL)
        .agg(
            pl.min(WASTE_DATE_COL).alias("min_date"),
            pl.max(WASTE_DATE_COL).alias("max_date"),
        )
    )

    location_service_days = (
        calendar.filter(pl.col("is_regular_service_day"))
        .select(pl.col("day").alias(WASTE_DATE_COL))
        .join(location_bounds, how="cross")
        .filter(pl.col(WASTE_DATE_COL).is_between(pl.col("min_date"), pl.col("max_date"), closed="both"))
        .drop("min_date", "max_date")
    )
    location_ingredient_units = pl.concat(
        [
            daily_waste.select(WASTE_LOCATION_COL, WASTE_INGREDIENT_COL, WASTE_UNIT_COL),
            exposure.select(WASTE_LOCATION_COL, WASTE_INGREDIENT_COL, WASTE_UNIT_COL),
        ],
        how="vertical",
    ).unique()

    full_grid = (
        location_service_days.join(location_ingredient_units, on=WASTE_LOCATION_COL, how="inner")
        .join(
            daily_waste,
            on=[WASTE_DATE_COL, WASTE_LOCATION_COL, WASTE_INGREDIENT_COL, WASTE_UNIT_COL],
            how="left",
        )
        .join(
            exposure,
            on=[WASTE_DATE_COL, WASTE_LOCATION_COL, WASTE_INGREDIENT_COL, WASTE_UNIT_COL],
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
        .group_by(
            WASTE_LOCATION_COL,
            WASTE_INGREDIENT_COL,
            WASTE_UNIT_COL,
            "ingredient_planned_in_available_recipes",
        )
        .agg(
            pl.len().alias("days"),
            pl.mean("daily_waste_quantity").alias("avg_waste_quantity"),
            pl.median("daily_waste_quantity").alias("median_waste_quantity"),
            pl.col("daily_waste_quantity").quantile(0.25).alias("q25_waste_quantity"),
            pl.col("daily_waste_quantity").quantile(0.75).alias("q75_waste_quantity"),
            pl.mean("planned_recipe_count").alias("avg_planned_recipe_count"),
        )
        .sort(
            WASTE_LOCATION_COL,
            WASTE_INGREDIENT_COL,
            WASTE_UNIT_COL,
            "ingredient_planned_in_available_recipes",
        )
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
    4. Waste concentration: which location/ingredient/unit series dominate waste?
    5. Recipe bridge: which location/ingredient/unit waste increases when planned?
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
                total_sales_analysis.calendar,
            )

    return ExtendedDemandExploration(
        total_sales_analysis=total_sales_analysis,
        daily_schedule_load=daily_schedule_load,
        schedule_sales_effect=schedule_sales_effect,
        category_schedule_effect=category_schedule_effect,
        daily_menu_mix=daily_menu_mix,
        daily_waste=daily_waste,
        ingredient_unit_waste=ingredient_unit_waste,
        waste_calendar_effect=waste_calendar_effect,
        recipe_ingredient_exposure=recipe_ingredient_exposure,
        ingredient_waste_when_planned=ingredient_waste_when_planned,
    )
