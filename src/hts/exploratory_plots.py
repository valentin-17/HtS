"""Plotting helpers for cafeteria exploratory demand analysis."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.container import BarContainer
import polars as pl

from .exploratory_analysis import (
    ACADEMIC_BUCKET_ORDER,
    ExtendedDemandExploration,
    SCHEDULE_PRESSURE_COL,
    SALES_CATEGORY_COL,
    SALES_DATE_COL,
    TotalSalesAnalysis,
    WEEKDAY_ORDER,
    WASTE_INGREDIENT_COL,
    WASTE_LOCATION_COL,
    WASTE_UNIT_COL,
)


SCHEDULE_LOAD_ORDER = ["no_scheduled_load", "low", "medium", "high"]


def _set_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")


def _first_present_column(df: pl.DataFrame, candidates: list[str]) -> str | None:
    return next((column for column in candidates if column in df.columns), None)


def _bar_label_if_possible(ax: plt.Axes, fmt: str = "%.0f") -> None:
    bar_container = next(
        (container for container in ax.containers if isinstance(container, BarContainer)),
        None,
    )
    if bar_container is not None:
        ax.bar_label(bar_container, fmt=fmt, padding=3)


def _iqr_error_kwargs(rows: list[dict], value_col: str, q25_col: str, q75_col: str) -> dict:
    if not rows or q25_col not in rows[0] or q75_col not in rows[0]:
        return {}
    lower = [
        max((row[value_col] or 0) - (row[q25_col] or row[value_col] or 0), 0)
        for row in rows
    ]
    upper = [
        max((row[q75_col] or row[value_col] or 0) - (row[value_col] or 0), 0)
        for row in rows
    ]
    return {
        "yerr": [lower, upper],
        "error_kw": {"elinewidth": 1.1, "capsize": 4, "capthick": 1.1},
    }


def _add_n_labels(ax: plt.Axes, rows: list[dict], count_col: str = "service_days") -> None:
    if not rows or count_col not in rows[0]:
        return
    ymin, ymax = ax.get_ylim()
    label_y = ymin + (ymax - ymin) * 0.02
    for index, row in enumerate(rows):
        ax.text(index, label_y, f"n={row[count_col]}", ha="center", va="bottom", fontsize=8)


def _axes_list(axes: plt.Axes | object) -> list[plt.Axes]:
    if hasattr(axes, "ravel"):
        return list(axes.ravel())
    return [axes]


def _top_n(df: pl.DataFrame, sort_col: str, n: int, descending: bool = True) -> pl.DataFrame:
    if df.is_empty() or sort_col not in df.columns:
        return df
    return df.sort(sort_col, descending=descending).head(n)


def _location_unit_rows(df: pl.DataFrame) -> list[dict]:
    if df.is_empty():
        return []
    return (
        df.select(WASTE_LOCATION_COL, WASTE_UNIT_COL)
        .unique()
        .sort(WASTE_LOCATION_COL, WASTE_UNIT_COL)
        .to_dicts()
    )


def plot_total_demand_over_time(daily_total_sales: pl.DataFrame) -> tuple[plt.Figure, plt.Axes]:
    """Plot daily total demand and a 10 service-day rolling average."""

    _set_style()
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
    ax.margins(x=0.01)
    ax.legend(frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig, ax


def plot_calendar_effects(analysis: TotalSalesAnalysis) -> tuple[plt.Figure, list[plt.Axes]]:
    """Plot weekday, season, semester, and academic-period effects."""

    _set_style()
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
            **_iqr_error_kwargs(rows, "avg_sales", "q25_sales", "q75_sales"),
        )
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("Average dishes sold")
        ax.tick_params(axis="x", rotation=30)
        _bar_label_if_possible(ax)
        _add_n_labels(ax, rows)

    fig.tight_layout()
    return fig, axes


def plot_weekday_lecture_vs_free(
    lecture_weekday_effect: pl.DataFrame,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot five weekday pairs comparing lecture and lecture-free periods."""

    _set_style()
    lookup = {
        (row["weekday"], row["period_type"]): row["avg_sales"]
        for row in lecture_weekday_effect.to_dicts()
    }
    lecture_values = [lookup.get((weekday, "Lecture period"), 0) for weekday in WEEKDAY_ORDER]
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

    for position, weekday in enumerate(WEEKDAY_ORDER):
        lecture_n = next(
            (
                row.get("service_days")
                for row in lecture_weekday_effect.to_dicts()
                if row["weekday"] == weekday and row["period_type"] == "Lecture period"
            ),
            0,
        )
        free_n = next(
            (
                row.get("service_days")
                for row in lecture_weekday_effect.to_dicts()
                if row["weekday"] == weekday and row["period_type"] == "Lecture-free period"
            ),
            0,
        )
        ax.text(position, 0, f"n={lecture_n}/{free_n}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    return fig, ax


def plot_category_share(category_effect: pl.DataFrame) -> tuple[plt.Figure, plt.Axes]:
    """Plot category share of annual demand as a horizontal bar chart."""

    _set_style()
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
    _bar_label_if_possible(ax, "%.1f%%")
    if rows and "service_days_offered" in rows[0]:
        max_share = max(row["sales_share_pct"] for row in rows) if rows else 0
        label_offset = max(max_share * 0.08, 1.0)
        for container, row in zip(ax.containers[0], rows):
            ax.text(
                container.get_width() + label_offset,
                container.get_y() + container.get_height() / 2,
                f"n={row['service_days_offered']}",
                va="center",
                ha="left",
                fontsize=8,
            )
        ax.set_xlim(right=max_share + label_offset * 4)
    fig.tight_layout()
    return fig, ax


def plot_total_sales_analysis(analysis: TotalSalesAnalysis) -> list[plt.Figure]:
    """Create all stakeholder-facing total-sales plots."""

    return [
        plot_total_demand_over_time(analysis.daily_total_sales)[0],
        plot_calendar_effects(analysis)[0],
        plot_weekday_lecture_vs_free(analysis.lecture_weekday_effect)[0],
        plot_category_share(analysis.category_effect)[0],
    ]


def plot_schedule_pressure(
    schedule_sales_effect: pl.DataFrame,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Plot sales against adjusted schedule pressure and compare proxy variants."""

    _set_style()
    summary = (
        schedule_sales_effect.group_by("schedule_load_bucket")
        .agg(
            pl.len().alias("service_days"),
            pl.mean("total_sales").alias("avg_sales"),
            pl.median("total_sales").alias("median_sales"),
            pl.col("total_sales").quantile(0.25).alias("q25_sales"),
            pl.col("total_sales").quantile(0.75).alias("q75_sales"),
        )
    )
    bucket_lookup = {
        row["schedule_load_bucket"]: row
        for row in summary.to_dicts()
    }
    bucket_rows = [
        bucket_lookup.get(
            bucket,
            {
                "schedule_load_bucket": bucket,
                "service_days": 0,
                "avg_sales": 0,
                "q25_sales": 0,
                "q75_sales": 0,
            },
        )
        for bucket in SCHEDULE_LOAD_ORDER
    ]
    bucket_values = [row["avg_sales"] for row in bucket_rows]
    rows = schedule_sales_effect.to_dicts()

    positive_proxy_days = schedule_sales_effect.filter(pl.col(SCHEDULE_PRESSURE_COL) > 0)
    proxy_source = positive_proxy_days if not positive_proxy_days.is_empty() else schedule_sales_effect
    proxy_means = proxy_source.select(
        pl.mean("registered_students_raw").alias("Raw appointment sum"),
        pl.mean("registered_students_course_deduplicated").alias("Course deduplicated"),
        pl.mean("registered_students_parallel_adjusted").alias("Parallel adjusted"),
        pl.mean("registered_students_time_weighted").alias("Raw time weighted"),
        pl.mean(SCHEDULE_PRESSURE_COL).alias("Parallel adjusted + time weighted"),
    ).to_dicts()[0]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    lecture_rows = [row for row in rows if row.get("is_lecture_day")]
    free_rows = [row for row in rows if not row.get("is_lecture_day")]
    axes[0].scatter(
        [row[SCHEDULE_PRESSURE_COL] for row in lecture_rows],
        [row["total_sales"] for row in lecture_rows],
        color="#4c78a8",
        alpha=0.7,
        label="Lecture days",
    )
    axes[0].scatter(
        [row[SCHEDULE_PRESSURE_COL] for row in free_rows],
        [row["total_sales"] for row in free_rows],
        color="#f58518",
        alpha=0.7,
        label="Lecture-free days",
    )
    axes[0].set_title("Daily Demand vs Adjusted Schedule Pressure")
    axes[0].set_xlabel("Parallel-adjusted time-weighted registered students")
    axes[0].set_ylabel("Dishes sold")
    axes[0].legend(frameon=False)

    axes[1].bar(
        SCHEDULE_LOAD_ORDER,
        bucket_values,
        color="#72b7b2",
        **_iqr_error_kwargs(bucket_rows, "avg_sales", "q25_sales", "q75_sales"),
    )
    axes[1].set_title("Average Demand by Schedule-Load Bucket")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Average dishes sold")
    axes[1].tick_params(axis="x", rotation=25)
    _bar_label_if_possible(axes[1])
    _add_n_labels(axes[1], bucket_rows)

    axes[2].barh(
        list(proxy_means.keys()),
        list(proxy_means.values()),
        color=["#bab0ac", "#9c755f", "#72b7b2", "#f58518", "#4c78a8"],
    )
    axes[2].set_title("Average Schedule Proxy Variants on Positive-Load Days")
    axes[2].set_xlabel("Registered-student pressure")
    axes[2].set_ylabel("")
    _bar_label_if_possible(axes[2], "%.0f")

    fig.tight_layout()
    return fig, list(axes)


def plot_category_schedule_effect(
    category_schedule_effect: pl.DataFrame,
    *,
    top_n_categories: int = 8,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot category-level average sales by adjusted schedule-load bucket."""

    _set_style()
    top_categories = (
        category_schedule_effect.group_by(SALES_CATEGORY_COL)
        .agg(pl.mean("avg_sales").alias("overall_avg_sales"))
        .sort("overall_avg_sales", descending=True)
        .head(top_n_categories)
        .get_column(SALES_CATEGORY_COL)
        .to_list()
    )
    data = category_schedule_effect.filter(pl.col(SALES_CATEGORY_COL).is_in(top_categories))
    lookup = {
        (row[SALES_CATEGORY_COL], row["schedule_load_bucket"]): row["avg_sales"]
        for row in data.to_dicts()
    }
    x_positions = list(range(len(top_categories)))
    bar_width = 0.18

    fig, ax = plt.subplots(figsize=(12, 5.5))
    colors = ["#bab0ac", "#4c78a8", "#f58518", "#54a24b"]
    for offset, bucket in enumerate(SCHEDULE_LOAD_ORDER):
        values = [lookup.get((category, bucket), 0) for category in top_categories]
        positions = [
            position + (offset - 1.5) * bar_width
            for position in x_positions
        ]
        ax.bar(positions, values, width=bar_width, label=bucket, color=colors[offset])

    ax.set_title("Category Demand by Adjusted Schedule-Load Bucket")
    ax.set_xlabel("")
    ax.set_ylabel("Average dishes sold")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(top_categories, rotation=30, ha="right")
    ax.legend(frameon=False)
    ax.margins(y=0.12)
    fig.tight_layout()
    return fig, ax


def plot_daily_menu_mix(daily_menu_mix: pl.DataFrame) -> tuple[plt.Figure, plt.Axes]:
    """Plot daily offer-count features from the meal plan."""

    _set_style()
    rows = daily_menu_mix.to_dicts()
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(
        [row["date"] for row in rows],
        [row["offered_dish_count"] for row in rows],
        color="#4c78a8",
        linewidth=1.6,
        label="Dishes",
    )
    ax.plot(
        [row["date"] for row in rows],
        [row["offered_category_count"] for row in rows],
        color="#f58518",
        linewidth=1.6,
        label="Categories",
    )
    ax.set_title("Daily Menu Breadth")
    ax.set_xlabel("")
    ax.set_ylabel("Offered count")
    ax.legend(frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig, ax


def plot_ingredient_unit_waste(
    ingredient_unit_waste: pl.DataFrame,
    *,
    top_n_series: int = 15,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Plot highest-waste ingredient series in separate panels by location and unit."""

    _set_style()
    panel_rows = _location_unit_rows(ingredient_unit_waste)
    if not panel_rows:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.set_axis_off()
        ax.set_title("No Waste Records Available")
        return fig, [ax]
    fig, axes_grid = plt.subplots(
        len(panel_rows),
        1,
        figsize=(11, max(3.5, 3.3 * len(panel_rows))),
    )
    axes = _axes_list(axes_grid)

    for ax, panel in zip(axes, panel_rows):
        location = panel[WASTE_LOCATION_COL]
        unit = panel[WASTE_UNIT_COL]
        data = _top_n(
            ingredient_unit_waste.filter(
                (pl.col(WASTE_LOCATION_COL) == location)
                & (pl.col(WASTE_UNIT_COL) == unit)
            ),
            "total_waste_quantity",
            top_n_series,
        )
        rows = data.sort("total_waste_quantity").to_dicts()
        ax.barh(
            [row[WASTE_INGREDIENT_COL] for row in rows],
            [row["total_waste_quantity"] for row in rows],
            color="#b279a2",
        )
        ax.set_title(f"Largest Waste Series - {location} ({unit})")
        ax.set_xlabel(f"Total waste quantity ({unit})")
        ax.set_ylabel("")
        _bar_label_if_possible(ax, "%.1f")

    fig.tight_layout()
    return fig, axes


def plot_waste_calendar_effect(
    waste_calendar_effect: pl.DataFrame,
    *,
    top_n_series: int = 10,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Plot average waste by academic period, location, and unit."""

    _set_style()
    panel_rows = _location_unit_rows(waste_calendar_effect)
    if not panel_rows:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.set_axis_off()
        ax.set_title("No Waste Calendar Effects Available")
        return fig, [ax]
    fig, axes_grid = plt.subplots(
        len(panel_rows),
        1,
        figsize=(11, max(3.5, 3.5 * len(panel_rows))),
    )
    axes = _axes_list(axes_grid)

    for ax, panel in zip(axes, panel_rows):
        location = panel[WASTE_LOCATION_COL]
        unit = panel[WASTE_UNIT_COL]
        panel_data = waste_calendar_effect.filter(
            (pl.col(WASTE_LOCATION_COL) == location)
            & (pl.col(WASTE_UNIT_COL) == unit)
        )
        selected = (
            panel_data.group_by(WASTE_INGREDIENT_COL)
            .agg(
                (pl.col("avg_waste_quantity") * pl.col("service_days"))
                .sum()
                .alias("estimated_total_waste")
            )
            .sort("estimated_total_waste", descending=True)
            .head(top_n_series)
            .get_column(WASTE_INGREDIENT_COL)
            .to_list()
        )
        selected_data = panel_data.filter(pl.col(WASTE_INGREDIENT_COL).is_in(selected))
        period_totals = (
            selected_data
            .group_by("academic_bucket")
            .agg(
                (pl.col("avg_waste_quantity") * pl.col("service_days"))
                .sum()
                .alias("estimated_total_waste"),
            )
        )
        period_days = (
            selected_data.group_by("academic_bucket", "weekday")
            .agg(pl.col("service_days").max().alias("weekday_service_days"))
            .group_by("academic_bucket")
            .agg(pl.sum("weekday_service_days").alias("service_days"))
        )
        rows = (
            period_totals.join(period_days, on="academic_bucket", how="left")
            .with_columns(
                (pl.col("estimated_total_waste") / pl.col("service_days")).alias(
                    "avg_waste_quantity"
                )
            )
            .join(
                pl.DataFrame(
                    {
                        "academic_bucket": ACADEMIC_BUCKET_ORDER,
                        "sort_order": list(range(len(ACADEMIC_BUCKET_ORDER))),
                    }
                ),
                on="academic_bucket",
                how="inner",
            )
            .sort("sort_order")
            .to_dicts()
        )
        ax.bar(
            [row["academic_bucket"] for row in rows],
            [row["avg_waste_quantity"] for row in rows],
            color="#59a14f",
        )
        ax.set_title(f"Average Waste by Academic Period - {location} ({unit})")
        ax.set_xlabel("")
        ax.set_ylabel(f"Average daily waste ({unit})")
        ax.tick_params(axis="x", rotation=25)
        _bar_label_if_possible(ax, "%.1f")

    fig.tight_layout()
    return fig, axes


def plot_ingredient_waste_when_planned(
    ingredient_waste_when_planned: pl.DataFrame,
    *,
    top_n_series: int = 12,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Compare waste when an ingredient appears in planned recipes, by location and unit."""

    _set_style()
    data = ingredient_waste_when_planned.with_columns(
        series_label=pl.concat_str(
            [pl.col(WASTE_INGREDIENT_COL), pl.lit(" / "), pl.col(WASTE_UNIT_COL)]
        )
    )
    panel_rows = _location_unit_rows(data)
    if not panel_rows:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.set_axis_off()
        ax.set_title("No Ingredient Planning Waste Comparison Available")
        return fig, [ax]
    fig, axes_grid = plt.subplots(
        len(panel_rows),
        1,
        figsize=(12, max(4, 3.9 * len(panel_rows))),
    )
    axes = _axes_list(axes_grid)

    for ax, panel in zip(axes, panel_rows):
        location = panel[WASTE_LOCATION_COL]
        unit = panel[WASTE_UNIT_COL]
        panel_data = data.filter(
            (pl.col(WASTE_LOCATION_COL) == location)
            & (pl.col(WASTE_UNIT_COL) == unit)
        )
        selected = (
            panel_data.group_by("series_label")
            .agg(pl.max("avg_waste_quantity").alias("max_avg_waste_quantity"))
            .sort("max_avg_waste_quantity", descending=True)
            .head(top_n_series)
            .get_column("series_label")
            .to_list()
        )
        filtered = panel_data.filter(pl.col("series_label").is_in(selected))
        series_labels = sorted(selected)
        lookup = {
            (row["series_label"], row["ingredient_planned_in_available_recipes"]): row[
                "avg_waste_quantity"
            ]
            for row in filtered.to_dicts()
        }
        x_positions = list(range(len(series_labels)))
        bar_width = 0.38
        not_planned = [
            lookup.get((series_label, False), 0)
            for series_label in series_labels
        ]
        planned = [
            lookup.get((series_label, True), 0)
            for series_label in series_labels
        ]
        ax.bar(
            [position - bar_width / 2 for position in x_positions],
            not_planned,
            width=bar_width,
            color="#bab0ac",
            label="Not planned in available recipes",
        )
        ax.bar(
            [position + bar_width / 2 for position in x_positions],
            planned,
            width=bar_width,
            color="#4c78a8",
            label="Planned in available recipes",
        )
        ax.set_title(f"Waste When Ingredient Appears in Planned Recipes - {location} ({unit})")
        ax.set_xlabel("")
        ax.set_ylabel(f"Average waste ({unit})")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(series_labels, rotation=35, ha="right")
        ax.legend(frameon=False)

    fig.tight_layout()
    return fig, axes


def named_total_sales_analysis_plots(
    analysis: TotalSalesAnalysis,
) -> list[tuple[str, plt.Figure]]:
    """Create named total-sales plots for deterministic disk output."""

    return [
        ("total_demand_over_time.png", plot_total_demand_over_time(analysis.daily_total_sales)[0]),
        ("calendar_effects.png", plot_calendar_effects(analysis)[0]),
        (
            "weekday_lecture_vs_free.png",
            plot_weekday_lecture_vs_free(analysis.lecture_weekday_effect)[0],
        ),
        ("category_share.png", plot_category_share(analysis.category_effect)[0]),
    ]


def named_extended_demand_exploration_plots(
    exploration: ExtendedDemandExploration,
) -> list[tuple[str, plt.Figure]]:
    """Create named stakeholder-facing plots for every available EDA output."""

    named_figures = named_total_sales_analysis_plots(exploration.total_sales_analysis)

    if exploration.schedule_sales_effect is not None:
        named_figures.append(
            ("schedule_pressure.png", plot_schedule_pressure(exploration.schedule_sales_effect)[0])
        )

    if exploration.category_schedule_effect is not None:
        named_figures.append(
            (
                "category_schedule_effect.png",
                plot_category_schedule_effect(exploration.category_schedule_effect)[0],
            )
        )

    if exploration.daily_menu_mix is not None:
        named_figures.append(("daily_menu_mix.png", plot_daily_menu_mix(exploration.daily_menu_mix)[0]))

    if exploration.ingredient_unit_waste is not None:
        named_figures.append(
            (
                "ingredient_unit_waste.png",
                plot_ingredient_unit_waste(exploration.ingredient_unit_waste)[0],
            )
        )

    if exploration.waste_calendar_effect is not None:
        named_figures.append(
            (
                "waste_calendar_effect.png",
                plot_waste_calendar_effect(exploration.waste_calendar_effect)[0],
            )
        )

    if exploration.ingredient_waste_when_planned is not None:
        named_figures.append(
            (
                "ingredient_waste_when_planned.png",
                plot_ingredient_waste_when_planned(
                    exploration.ingredient_waste_when_planned
                )[0],
            )
        )

    return named_figures


def plot_extended_demand_exploration(
    exploration: ExtendedDemandExploration,
) -> list[plt.Figure]:
    """Create available stakeholder-facing plots for the extended EDA."""

    return [
        figure
        for _, figure in named_extended_demand_exploration_plots(exploration)
    ]
