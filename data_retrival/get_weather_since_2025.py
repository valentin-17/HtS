from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry


LATITUDE = 49.7557
LONGITUDE = 6.6394
START_DATE = date(2025, 1, 1)
END_DATE = date.today()
ROLLING_WINDOW_DAYS = 5

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "rain",
    "cloud_cover_high",
    "cloud_cover_mid",
    "cloud_cover_low",
    "cloud_cover",
    "precipitation",
    "apparent_temperature",
]


def build_openmeteo_client() -> openmeteo_requests.Client:
    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)


def fetch_hourly_weather() -> pd.DataFrame:
    openmeteo = build_openmeteo_client()
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": START_DATE.isoformat(),
        "end_date": END_DATE.isoformat(),
        "hourly": HOURLY_VARIABLES,
        "timezone": "Europe/Berlin",
    }
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation: {response.Elevation()} m asl")
    print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")
    print(f"History window: {START_DATE.isoformat()} to {END_DATE.isoformat()}")

    hourly = response.Hourly()
    data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        )
    }

    for i, var_name in enumerate(HOURLY_VARIABLES):
        data[var_name] = hourly.Variables(i).ValuesAsNumpy()

    df = pd.DataFrame(data)
    df["date"] = df["date"].dt.tz_convert("Europe/Berlin")
    return df


def build_weekday_daily_frame(hourly_df: pd.DataFrame) -> pd.DataFrame:
    daily = (
        hourly_df.set_index("date")
        .resample("D")
        .agg(
            temperature_2m=("temperature_2m", "mean"),
            apparent_temperature=("apparent_temperature", "mean"),
            precipitation=("precipitation", "sum"),
            rain=("rain", "sum"),
            relative_humidity_2m=("relative_humidity_2m", "mean"),
            cloud_cover=("cloud_cover", "mean"),
        )
        .reset_index()
    )
    daily["is_weekend"] = daily["date"].dt.weekday >= 5
    service_days = daily.loc[~daily["is_weekend"]].copy()
    service_days["apparent_temperature_rolling_avg"] = service_days[
        "apparent_temperature"
    ].rolling(window=ROLLING_WINDOW_DAYS, min_periods=3).mean()
    service_days["rain_rolling_avg"] = service_days["rain"].rolling(
        window=ROLLING_WINDOW_DAYS, min_periods=3
    ).mean()
    service_days["precipitation_rolling_avg"] = service_days["precipitation"].rolling(
        window=ROLLING_WINDOW_DAYS, min_periods=3
    ).mean()
    service_days["cloud_cover_rolling_avg"] = service_days["cloud_cover"].rolling(
        window=ROLLING_WINDOW_DAYS, min_periods=3
    ).mean()
    return service_days


def plot_weekday_weather(daily_weekdays: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

    axes[0].plot(
        daily_weekdays["date"],
        daily_weekdays["apparent_temperature"],
        color="#9aa0a6",
        linewidth=1.0,
        alpha=0.5,
        label="Daily mean apparent temperature",
    )
    axes[0].plot(
        daily_weekdays["date"],
        daily_weekdays["apparent_temperature_rolling_avg"],
        color="#1f77b4",
        linewidth=2.2,
        label=f"{ROLLING_WINDOW_DAYS} service-day average",
    )
    axes[0].set_title("Weekday Weather Trend (No Weekends)")
    axes[0].set_ylabel("Apparent temperature (°C)")
    axes[0].legend(frameon=False)

    axes[1].plot(
        daily_weekdays["date"],
        daily_weekdays["precipitation"],
        color="#9aa0a6",
        linewidth=1.0,
        alpha=0.5,
        label="Daily precipitation amount",
    )
    axes[1].plot(
        daily_weekdays["date"],
        daily_weekdays["precipitation_rolling_avg"],
        color="#4c78a8",
        linewidth=2.2,
        label=f"{ROLLING_WINDOW_DAYS} service-day average",
    )
    axes[1].set_ylabel("Precipitation (mm/day)")
    axes[1].set_title("Weekday Precipitation Trend (No Weekends)")
    axes[1].legend(frameon=False)

    axes[2].plot(
        daily_weekdays["date"],
        daily_weekdays["cloud_cover"],
        color="#9aa0a6",
        linewidth=1.0,
        alpha=0.5,
        label="Daily mean cloud cover",
    )
    axes[2].plot(
        daily_weekdays["date"],
        daily_weekdays["cloud_cover_rolling_avg"],
        color="#4c78a8",
        linewidth=2.2,
        label=f"{ROLLING_WINDOW_DAYS} service-day average",
    )
    axes[2].set_ylabel("Cloud cover (%)")
    axes[2].set_title("Weekday Cloud Cover Trend (No Weekends)")
    axes[2].set_xlabel("")
    axes[2].legend(frameon=False)

    fig.autofmt_xdate()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    hourly_df = fetch_hourly_weather()
    weekday_daily_df = build_weekday_daily_frame(hourly_df)

    output_root = Path(__file__).resolve().parent
    csv_output = output_root / "data" / "weather_weekday_daily_since_2025.csv"
    plot_output = output_root / "plots" / "weather_weekday_since_2025.png"

    csv_output.parent.mkdir(parents=True, exist_ok=True)
    weekday_daily_df.to_csv(csv_output, index=False)
    plot_weekday_weather(weekday_daily_df, plot_output)

    print(f"\nSaved weekday weather data to: {csv_output}")
    print(f"Saved weather visualization to: {plot_output}")
    print("\nWeekday daily data preview:")
    print(weekday_daily_df.head())


if __name__ == "__main__":
    main()
