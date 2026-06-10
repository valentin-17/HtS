"""SQLAlchemy-Modelle für Verkaufszahlen und Tages-Features."""

from __future__ import annotations

from datetime import date

from sqlalchemy import Date, Float, Integer, String, Boolean, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Sale(Base):
    """Eine Verkaufszeile: Prod.Art / Rezeptur / Ist-Menge an einem Tag."""

    __tablename__ = "sales"
    __table_args__ = (
        UniqueConstraint("date", "prod_art", "rezeptur", name="uq_sale_day_art_rezeptur"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[date] = mapped_column(Date, index=True, nullable=False)
    prod_art: Mapped[str] = mapped_column(String, nullable=False)
    rezeptur: Mapped[str] = mapped_column(String, nullable=False, default="")
    ist_menge: Mapped[int | None] = mapped_column(Integer, nullable=True)


class DayFeature(Base):
    """Kalender- und Wetter-Features pro Tag (spiegelt final_feature_df)."""

    __tablename__ = "day_features"

    date: Mapped[date] = mapped_column(Date, primary_key=True)
    weekday: Mapped[str | None] = mapped_column(String, nullable=True)
    kw: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_weekend: Mapped[bool] = mapped_column(Boolean, default=False)
    semester: Mapped[str | None] = mapped_column(String, nullable=True)
    academic_bucket: Mapped[str | None] = mapped_column(String, nullable=True)
    is_public_holiday: Mapped[bool] = mapped_column(Boolean, default=False)
    public_holiday_name: Mapped[str | None] = mapped_column(String, nullable=True)
    is_university_closure: Mapped[bool] = mapped_column(Boolean, default=False)

    # Wetterspalten (Open-Meteo)
    sunshine_duration: Mapped[float | None] = mapped_column(Float, nullable=True)
    precipitation_hours: Mapped[float | None] = mapped_column(Float, nullable=True)
    apparent_temperature_mean: Mapped[float | None] = mapped_column(Float, nullable=True)
    precipitation_sum: Mapped[float | None] = mapped_column(Float, nullable=True)
    weather_category: Mapped[str | None] = mapped_column(String, nullable=True)
