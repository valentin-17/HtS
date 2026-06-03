# Hack the Snack

Focused cafeteria visitor-demand prediction for the one usable university
cafeteria location.

The real datasets are under NDA and must never be stored inside this workspace.
Use a private data directory outside the repository, for example:

```text
C:\Users\<user>\HtS-data
```

Configure that location through the `HTS_DATA_ROOT` environment variable in a
local `.env` file.

## Current Scope

The repository keeps only code and documentation for these inputs:

- daily sales by meal category and dish, used as the visitor-count proxy
- menu attributes, currently vegetarian, pork, and price
- daily meal plan for the selected cafeteria location
- rough yearly enrollment counts, used only as a campus-population upper bound
- academic calendar features
- weather features

Everything outside this prediction task, including waste and recipe-ingredient
exposure analysis, has been removed.

## Code

- `src/hts/exploratory_analysis.py`: public import facade for the prediction workflow
- `src/hts/demand_*.py`: small modules for calendar, normalization, features, models, intervals, and validation
- `explorative_analysis.ipynb`: compact example notebook using mock data only
- `src/hts/probabilistic_forecasting.py`: fresh daily probabilistic sales forecast for total cafeteria dish sales

## Daily Probabilistic Forecast

The current forecasting entry point predicts total dish sales for the next five
regular service days and returns predictive quantiles, not confidence
intervals:

```python
from hts import (
    backtest_probabilistic_forecaster,
    forecast_next_service_days,
    load_feature_frame,
)

history = load_feature_frame(r"C:\private-data\final_feature_df.csv")
future = load_feature_frame(r"C:\private-data\future_feature_df_no_target.csv")

forecast = forecast_next_service_days(history, future, horizon=5)
validation = backtest_probabilistic_forecaster(history)
```

Forecast columns include `p05`, `p10`, `p50`, `p90`, `p95`, `interval_80`,
`interval_90`, empirical baseline quantiles, and the selected similar days used
as context. Training uses only rows where `is_regular_service_day` is true and
`sales` is present.
