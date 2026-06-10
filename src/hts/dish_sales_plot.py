"""
Run with: streamlit run src/hts_nda/dish_sales_plot.py
"""

import streamlit as st
import plotly.express as px
import polars as pl

st.set_page_config(
    page_title='Mensa Data Dashboard',
    layout='wide',
)

st.title('Mensa Data Dashboard')

segment_colors = {
    'Konsistent starke Verkäufe': '#53A12E',
    'Konsistent schwache Verkäufe': '#D02525',
    'Beliebt aber volatil': '#3F92D2',
    'Niedrig und unvorhersehbar': '#F3B700',
    'Mittlerer Bereich': '#9E9E9E',
}

meal_categories = [
    'Menü 1',
    'Menü 2',
    'Eintopf Groß',
    'Dessert',
    'Joker',
    'Salatteller',
    'Sweet Joker',
    'Tellergericht Pasta Cor.',
]


@st.cache_data
def load_sales_processed() -> pl.DataFrame:
    prepped_data = pl.read_csv('data/processed/final_feature_df.csv',try_parse_dates=True,)
    sales_processed = pl.read_csv('data/processed/sales_2025_processed_from_csv.csv',try_parse_dates=True,)

    return sales_processed.join(
        prepped_data.select(['date', 'academic_bucket']),
        on='date',
        how='left',
    )

def compute_dish_stats(
    sales_processed: pl.DataFrame,
    selected_academic_buckets: list[str],
    min_days_sold: int,
    min_total_sold_quantile: float,
) -> pl.DataFrame:
    df = sales_processed

    if selected_academic_buckets:
        df = df.filter(pl.col('academic_bucket').is_in(selected_academic_buckets))

    return (
        df
        .filter(pl.col('meal_category').is_in(meal_categories))
        .group_by('meal_name', 'meal_category')
        .agg([
            pl.col('sales').sum().alias('total_sold'),
            pl.col('sales').mean().alias('avg_sold'),
            pl.col('sales').std().alias('std_sold'),
            pl.col('date').n_unique().alias('days_sold'),
        ])
        .with_columns(
            (pl.col('std_sold') / pl.col('avg_sold')).alias('cv')
        )
        .with_columns(
            (
                pl.col('avg_sold') / (1 + pl.col('cv'))
            ).alias('consistency_score')
        )
        .filter(
            (pl.col('days_sold') >= min_days_sold) &
            (pl.col('total_sold') >= pl.col('total_sold').quantile(min_total_sold_quantile))
        )
    )

def add_segments(
    dish_stats: pl.DataFrame,
    sales_cutoff_quantiles: tuple[float, float],
    cv_cutoff_quantiles: tuple[float, float],
) -> tuple[pl.DataFrame, dict]:
    cutoffs = dish_stats.select([
        pl.col('avg_sold').quantile(sales_cutoff_quantiles[1]).alias('high_sales_cutoff'),
        pl.col('avg_sold').quantile(sales_cutoff_quantiles[0]).alias('low_sales_cutoff'),
        pl.col('cv').quantile(cv_cutoff_quantiles[0]).alias('stable_cutoff'),
        pl.col('cv').quantile(cv_cutoff_quantiles[1]).alias('unstable_cutoff'),
    ]).row(0, named=True)

    dish_segments = (
        dish_stats
        .with_columns(
            pl.when(
                (pl.col('avg_sold') >= cutoffs['high_sales_cutoff']) &
                (pl.col('cv') <= cutoffs['stable_cutoff'])
            )
            .then(pl.lit('Konsistent starke Verkäufe'))

            .when(
                (pl.col('avg_sold') <= cutoffs['low_sales_cutoff']) &
                (pl.col('cv') <= cutoffs['stable_cutoff'])
            )
            .then(pl.lit('Konsistent schwache Verkäufe'))

            .when(
                (pl.col('avg_sold') >= cutoffs['high_sales_cutoff']) &
                (pl.col('cv') >= cutoffs['unstable_cutoff'])
            )
            .then(pl.lit('Beliebt aber volatil'))

            .when(
                (pl.col('avg_sold') <= cutoffs['low_sales_cutoff']) &
                (pl.col('cv') >= cutoffs['unstable_cutoff'])
            )
            .then(pl.lit('Niedrig und unvorhersehbar'))

            .otherwise(pl.lit('Mittlerer Bereich'))
            .alias('segment')
        )
    )

    return dish_segments, cutoffs

### Main Script

sales_processed = load_sales_processed()

academic_buckets = sorted(
    sales_processed
    .select('academic_bucket')
    .drop_nulls()
    .unique()
    .to_series()
    .to_list()
)

use_lecture = st.checkbox(
    'Vorlesungszeit',
    value=True,
)

use_lecture_free = st.checkbox(
    'Vorlesungsfrei',
    value=True,
)

selected_academic_buckets = []

if use_lecture:
    selected_academic_buckets.append('lecture')

if use_lecture_free:
    selected_academic_buckets.append('lecture_free')

min_days_sold = st.slider(
    'Untergrenze Verkaufstage (absolut)',
    min_value=1,
    max_value=30,
    value=5,
    step=1,
)

min_total_sold_quantile = st.slider(
    'Untergrenze Verkäufe Gesamt (Quantil)',
    min_value=0.0,
    max_value=0.9,
    value=0.1,
    step=0.05,
)

sales_cutoff_quantiles = st.slider(
    'Grenzen Durchschnittliche Verkäufe/Tag (Quantile)',
    min_value=0.0,
    max_value=1.0,
    value=(0.25, 0.75),
    step=0.01,
)

cv_cutoff_quantiles = st.slider(
    'Grenzen Koeffizient der Varianz (Quantile)',
    min_value=0.0,
    max_value=1.0,
    value=(0.25, 0.75),
    step=0.01,
)

dish_stats = compute_dish_stats(
    sales_processed=sales_processed,
    selected_academic_buckets=selected_academic_buckets,
    min_days_sold=min_days_sold,
    min_total_sold_quantile=min_total_sold_quantile,
)

if dish_stats.is_empty():
    st.warning('Keine Daten für die aktuelle Auswahl.')
    st.stop()

dish_segments, cutoffs = add_segments(
    dish_stats=dish_stats,
    sales_cutoff_quantiles=sales_cutoff_quantiles,
    cv_cutoff_quantiles=cv_cutoff_quantiles,
)

plot_df = (
    dish_segments
    .rename({
        'meal_name': 'Gericht',
        'meal_category': 'Kategorie',
        'total_sold': 'Verkäufe Gesamt',
        'avg_sold': 'Durchschnittliche Verkäufe/Tag',
        'cv': 'Koeffizient der Varianz',
        'days_sold': 'An Tagen verkauft',
        'segment': 'Segment',
    })
    .to_pandas()
)

segments = sorted(plot_df['Segment'].dropna().unique())
categories = sorted(plot_df['Kategorie'].dropna().unique())

selected_segments = st.multiselect(
    'Segment',
    options=segments,
    default=segments,
)

selected_categories = st.multiselect(
    'Kategorie',
    options=categories,
    default=categories,
)

filtered_df = plot_df[
    plot_df['Segment'].isin(selected_segments) &
    plot_df['Kategorie'].isin(selected_categories)
]

fig = px.scatter(
    filtered_df,
    x='Durchschnittliche Verkäufe/Tag',
    y='Koeffizient der Varianz',
    color='Segment',
    size='Verkäufe Gesamt',
    hover_name='Gericht',
    hover_data={
        'Kategorie': True,
        'Verkäufe Gesamt': True,
        'Durchschnittliche Verkäufe/Tag': ':.2f',
        'Koeffizient der Varianz': ':.2f',
        'An Tagen verkauft': True,
        'Segment': False,
    },
    color_discrete_map=segment_colors,
    size_max=30,
    title='Verkaufsstabilität von Gerichten'
)

fig.update_yaxes(range=[0, 1])

fig.update_layout(
    width=1200,
    height=800,
    template='plotly_white',
    xaxis_title='Durchschnittliche Verkäufe/Tag',
    yaxis_title='Koeffizient der Varianz (CV)',
    legend_title='Segment',
    legend=dict(x=0.98, y=0.98, xanchor="right", yanchor="top")
)

fig.add_vline(
    x=cutoffs['low_sales_cutoff'],
    line_dash='dash',
    line_color='gray',
    annotation_text='Wenige Verkäufe ',
    annotation_position='top left'
)

fig.add_vline(
    x=cutoffs['high_sales_cutoff'],
    line_dash='dash',
    line_color='gray',
    annotation_text=' Viele Verkäufe',
    annotation_position='top right'
)

fig.add_hline(
    y=cutoffs['stable_cutoff'],
    line_dash='dash',
    line_color='gray',
    annotation_text='Stabiler Bereich',
    annotation_position='bottom right'
)

fig.add_hline(
    y=cutoffs['unstable_cutoff'],
    line_dash='dash',
    line_color='gray',
    annotation_text='Instabiler Bereich',
    annotation_position='top right'
)

st.plotly_chart(fig, width='content')
