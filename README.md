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

## Category-Day Prediction Workflow

The first prediction-ready feature set uses one row per open cafeteria day,
meal category, and planned dish. Build it with:

```python
from hts.exploratory_analysis import (
    build_calendar,
    build_category_day_modeling_frame,
    build_prediction_frame,
    fit_category_weekday_baseline_model,
    add_point_predictions,
    build_residual_bands,
    predict_category_sales_range,
)

calendar, _, _ = build_calendar(2025)
frame = build_category_day_modeling_frame(
    sales,
    meal_plan,
    menu_attributes,
    calendar,
    weather=weather,
    location="selected cafeteria",
)
```

The workflow keeps dish names for reporting and attribute joins, but the first
model uses category, calendar, weather, price, vegetarian/vegan, pork, and
leakage-safe historical category-demand features.
