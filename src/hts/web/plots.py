"""Plotly-Visualisierungen der Verkaufszahlen für den Reiter 'Datenanalyse'.

Datengrundlage ist die Datenbank (Sale + DayFeature). Die Auswertungen entsprechen
denen aus ``data_prep.ipynb`` / ``dish_sales_plot.py``.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from . import config
from .db import session_scope
from .models import DayFeature, Sale


WEEKDAY_ORDER = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag"]


def _load_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lädt Sale- und DayFeature-Daten als pandas-DataFrames."""
    with session_scope() as session:
        sales = pd.DataFrame(
            session.query(
                Sale.date, Sale.prod_art, Sale.rezeptur, Sale.ist_menge
            ).all(),
            columns=["date", "prod_art", "rezeptur", "ist_menge"],
        )
        features = pd.DataFrame(
            session.query(
                DayFeature.date,
                DayFeature.weekday,
                DayFeature.apparent_temperature_mean,
                DayFeature.semester,
                DayFeature.is_weekend,
                DayFeature.is_university_closure,
            ).all(),
            columns=[
                "date",
                "weekday",
                "apparent_temperature_mean",
                "semester",
                "is_weekend",
                "is_university_closure",
            ],
        )
    return sales, features


def _daily_sales(sales: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    daily = (
        sales.dropna(subset=["ist_menge"])
        .groupby("date", as_index=False)["ist_menge"]
        .sum()
        .rename(columns={"ist_menge": "sales"})
    )
    merged = daily.merge(features, on="date", how="left")
    # Reguläre Servicetage: Mo–Fr und keine Schließung.
    return merged[(~merged["is_weekend"].fillna(False)) & (~merged["is_university_closure"].fillna(False))]


def _fig_html(fig: go.Figure) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _plot_weekday(daily: pd.DataFrame) -> go.Figure:
    df = daily.dropna(subset=["weekday"])
    agg = df.groupby("weekday", as_index=False)["sales"].mean()
    agg["weekday"] = pd.Categorical(agg["weekday"], categories=WEEKDAY_ORDER, ordered=True)
    agg = agg.sort_values("weekday")
    fig = px.bar(
        agg,
        x="weekday",
        y="sales",
        title="Durchschnittliche Verkäufe nach Wochentag",
        labels={"weekday": "Wochentag", "sales": "Verkäufe"},
        color_discrete_sequence=[config.COLORS["blau"]],
    )
    fig.update_layout(template="plotly_white")
    return fig


def _plot_temperature(daily: pd.DataFrame) -> go.Figure:
    df = daily.dropna(subset=["apparent_temperature_mean", "sales"])
    fig = px.scatter(
        df,
        x="apparent_temperature_mean",
        y="sales",
        color="semester",
        trendline="ols",
        title="Verkäufe nach gefühlter Temperatur",
        labels={
            "apparent_temperature_mean": "Gefühlte mittlere Temperatur (°C)",
            "sales": "Verkäufe",
            "semester": "Semester",
        },
    )
    fig.update_layout(template="plotly_white")
    return fig


def _plot_stability(sales: pd.DataFrame) -> go.Figure:
    df = sales.dropna(subset=["ist_menge"])
    df = df[df["prod_art"].isin(config.MEAL_CATEGORY_ORDER)]
    stats = (
        df.groupby(["rezeptur", "prod_art"], as_index=False)
        .agg(
            total_sold=("ist_menge", "sum"),
            avg_sold=("ist_menge", "mean"),
            std_sold=("ist_menge", "std"),
            days_sold=("date", "nunique"),
        )
    )
    stats["cv"] = stats["std_sold"] / stats["avg_sold"]
    stats = stats.dropna(subset=["cv"])
    if stats.empty:
        return go.Figure()

    stats = stats[
        (stats["days_sold"] >= 5)
        & (stats["total_sold"] >= stats["total_sold"].quantile(0.1))
    ]
    if stats.empty:
        return go.Figure()

    high = stats["avg_sold"].quantile(0.75)
    low = stats["avg_sold"].quantile(0.25)
    stable = stats["cv"].quantile(0.25)
    unstable = stats["cv"].quantile(0.75)

    def segment(row):
        if row["avg_sold"] >= high and row["cv"] <= stable:
            return "Konsistent starke Verkäufe"
        if row["avg_sold"] <= low and row["cv"] <= stable:
            return "Konsistent schwache Verkäufe"
        if row["avg_sold"] >= high and row["cv"] >= unstable:
            return "Starke Verkäufe aber volatil"
        if row["avg_sold"] <= low and row["cv"] >= unstable:
            return "Niedrige Verkäufe und unvorhersehbar"
        return "Mittlerer Bereich"

    stats["Segment"] = stats.apply(segment, axis=1)
    fig = px.scatter(
        stats,
        x="avg_sold",
        y="cv",
        color="Segment",
        size="total_sold",
        hover_name="rezeptur",
        hover_data={"prod_art": True, "days_sold": True},
        color_discrete_map=config.SEGMENT_COLORS,
        size_max=30,
        title="Verkaufsstabilität der Gerichte",
        labels={"avg_sold": "Durchschnittliche Verkäufe/Tag", "cv": "Koeffizient der Varianz"},
    )
    fig.update_yaxes(range=[0, 1])
    fig.update_layout(template="plotly_white")
    return fig


def build_plots() -> list[dict]:
    """Erzeugt alle Plots als HTML-Fragmente. Leere Datenlage wird abgefangen."""
    sales, features = _load_frames()
    if sales.empty:
        return []

    daily = _daily_sales(sales, features)

    plots: list[dict] = []
    if not daily.empty:
        plots.append({"title": "Wochentag", "html": _fig_html(_plot_weekday(daily))})
        if daily["apparent_temperature_mean"].notna().any():
            plots.append({"title": "Temperatur", "html": _fig_html(_plot_temperature(daily))})

    stability = _plot_stability(sales)
    if stability.data:
        plots.append({"title": "Verkaufsstabilität", "html": _fig_html(stability)})

    return plots
