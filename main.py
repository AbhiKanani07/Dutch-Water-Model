from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parent
PLOTS_DIR = PROJECT_ROOT / "plots"

DATA_PATH = Path(r"c:\Users\arkan\Downloads\table__82883ENG.csv")
OECD_PATH = PROJECT_ROOT / "oecd_indicators.csv"


def ensure_plots_dir() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_water_use_data(path: Path) -> pd.DataFrame:
    # Treat "." as missing and "," as thousands separator (values like "1,235.6")
    df = pd.read_csv(path, na_values=".", thousands=",")

    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[()]", "", regex=True)
    )

    year_series = df["periods"].astype(str).str.extract(r"(\d{4})")[0]
    df["year"] = pd.to_numeric(year_series, errors="coerce")
    df = df[df["year"].notna()].copy()
    df["year"] = df["year"].astype("int64")
    df["sector"] = df["waterusers"]

    col_tap_total = [c for c in df.columns if "tap_water_total_use_of_tap_water" in c][0]
    col_ground_total = [
        c for c in df.columns if "groundwater_total_use_of_groundwater" in c
    ][0]
    col_surface_total = [
        c for c in df.columns if "surface_water_total_use_of_surface_water" in c
    ][0]

    # Ensure the three main water-use columns are numeric
    for col in (col_tap_total, col_ground_total, col_surface_total):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["total_abstracted_mln_m3"] = df[
        [col_tap_total, col_ground_total, col_surface_total]
    ].sum(axis=1, min_count=1)

    df["share_tap"] = df[col_tap_total] / df["total_abstracted_mln_m3"]
    df["share_groundwater"] = df[col_ground_total] / df["total_abstracted_mln_m3"]
    df["share_surface_water"] = df[col_surface_total] / df["total_abstracted_mln_m3"]

    return df


def basic_eda(df: pd.DataFrame) -> None:
    print("Shape:", df.shape)
    print("\nColumns:", list(df.columns))
    print("\nMissing values per column:")
    print(df.isna().sum())

    print("\nNumeric summary (main water-use variables):")
    cols_summary = [
        c
        for c in df.columns
        if any(
            key in c
            for key in [
                "tap_water_total_use_of_tap_water",
                "groundwater_total_use_of_groundwater",
                "surface_water_total_use_of_surface_water",
                "total_abstracted_mln_m3",
            ]
        )
    ]
    print(df[cols_summary].describe())

    overall_mean = np.nanmean(df["total_abstracted_mln_m3"].to_numpy())
    print("\nOverall mean total abstraction (mln m3):", overall_mean)

    total_economy = df[df["sector"] == "Total Dutch economy"]
    group_cols = [
        c
        for c in cols_summary
        if c != "total_abstracted_mln_m3"
    ]
    yearly_totals = total_economy.groupby("year")[group_cols].sum()
    print("\nTotal Dutch economy yearly water use (mln m3):")
    print(yearly_totals)

    latest_year = df["year"].max()
    latest = df[df["year"] == latest_year]
    sector_totals = (
        latest.groupby("sector")["total_abstracted_mln_m3"]
        .sum()
        .sort_values(ascending=False)
    )
    print(f"\nTop water-using sectors in {latest_year}:")
    print(sector_totals.head(10))


def plot_trends(df: pd.DataFrame) -> None:
    ensure_plots_dir()
    total_economy = df[df["sector"] == "Total Dutch economy"]

    cols_trend = [
        c
        for c in df.columns
        if any(
            key in c
            for key in [
                "tap_water_total_use_of_tap_water",
                "groundwater_total_use_of_groundwater",
                "surface_water_total_use_of_surface_water",
            ]
        )
    ]

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 5))

    for col in cols_trend:
        plt.plot(total_economy["year"], total_economy[col], marker="o", label=col)

    plt.title("Water use by source - Total Dutch economy")
    plt.xlabel("Year")
    plt.ylabel("Volume (million m3)")
    plt.legend()
    plt.tight_layout()
    output_path = PLOTS_DIR / "water_trends.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def forecast_total_economy(df: pd.DataFrame, horizon_years: int = 5) -> pd.DataFrame:
    """Simple linear-trend forecast for the Total Dutch economy by source."""
    total_economy = df[df["sector"] == "Total Dutch economy"].copy()
    cols_trend = [
        c
        for c in df.columns
        if any(
            key in c
            for key in [
                "tap_water_total_use_of_tap_water",
                "groundwater_total_use_of_groundwater",
                "surface_water_total_use_of_surface_water",
            ]
        )
    ]

    last_year = int(total_economy["year"].max())
    future_years = np.arange(last_year + 1, last_year + horizon_years + 1)

    records: list[dict] = []
    for col in cols_trend:
        valid = total_economy[["year", col]].dropna()
        if len(valid) < 2:
            continue
        slope, intercept = np.polyfit(valid["year"], valid[col], 1)
        preds = slope * future_years + intercept
        for year, pred in zip(future_years, preds, strict=False):
            records.append(
                {
                    "year": int(year),
                    "source": col,
                    "predicted_mln_m3": float(pred),
                }
            )

    return pd.DataFrame.from_records(records)


def plot_forecast_total_economy(df: pd.DataFrame, forecast_df: pd.DataFrame) -> None:
    ensure_plots_dir()
    total_economy = df[df["sector"] == "Total Dutch economy"].copy()

    cols_trend = [
        c
        for c in df.columns
        if any(
            key in c
            for key in [
                "tap_water_total_use_of_tap_water",
                "groundwater_total_use_of_groundwater",
                "surface_water_total_use_of_surface_water",
            ]
        )
    ]

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 5))

    for col in cols_trend:
        plt.plot(
            total_economy["year"],
            total_economy[col],
            marker="o",
            label=f"{col} (actual)",
        )

        subset = forecast_df[forecast_df["source"] == col]
        if not subset.empty:
            plt.plot(
                subset["year"],
                subset["predicted_mln_m3"],
                linestyle="--",
                marker="o",
                label=f"{col} (forecast)",
            )

    plt.title("Water use by source - Forecast (Total Dutch economy)")
    plt.xlabel("Year")
    plt.ylabel("Volume (million m3)")
    plt.legend()
    plt.tight_layout()
    output_path = PLOTS_DIR / "forecast_total_economy.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved forecast plot to", output_path)


def plot_households_vs_activities(df: pd.DataFrame) -> None:
    ensure_plots_dir()
    sectors_of_interest = ["Private households", "A-U All economic activities"]
    subset = df[df["sector"].isin(sectors_of_interest)]

    yearly = (
        subset.groupby(["year", "sector"])["total_abstracted_mln_m3"]
        .sum()
        .reset_index()
    )

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 5))
    sns.lineplot(
        data=yearly,
        x="year",
        y="total_abstracted_mln_m3",
        hue="sector",
        marker="o",
    )
    plt.title("Total abstraction: households vs all activities")
    plt.xlabel("Year")
    plt.ylabel("Volume (million m3)")
    plt.tight_layout()
    output_path = PLOTS_DIR / "households_vs_activities.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_fresh_vs_salt_surface(df: pd.DataFrame) -> None:
    ensure_plots_dir()
    total_economy = df[df["sector"] == "Total Dutch economy"]

    col_fresh = [
        c for c in df.columns if "total_use_of_fresh_surface_water" in c
    ][0]
    col_salt = [
        c for c in df.columns if "total_use_of_salt_surface_water" in c
    ][0]

    yearly = (
        total_economy.groupby("year")[[col_fresh, col_salt]]
        .sum()
        .reset_index()
    )

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 5))
    plt.plot(yearly["year"], yearly[col_fresh], marker="o", label="Fresh surface water")
    plt.plot(yearly["year"], yearly[col_salt], marker="o", label="Salt surface water")
    plt.title("Fresh vs salt surface water - Total Dutch economy")
    plt.xlabel("Year")
    plt.ylabel("Volume (million m3)")
    plt.legend()
    plt.tight_layout()
    output_path = PLOTS_DIR / "fresh_vs_salt_surface.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def load_oecd_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    if "year" not in df.columns:
        candidates = [c for c in df.columns if "year" in c]
        if candidates:
            df = df.rename(columns={candidates[0]: "year"})

    df["year"] = df["year"].astype(int)
    return df


def merge_with_oecd(water_df: pd.DataFrame, oecd_df: pd.DataFrame) -> pd.DataFrame:
    return pd.merge(water_df, oecd_df, on="year", how="left")


def download_csv(url: str, out_path: Path) -> Path:
    import requests

    response = requests.get(url)
    response.raise_for_status()
    out_path.write_bytes(response.content)
    return out_path


def prompt_horizon(default: int = 5) -> int:
    try:
        raw = input(f"How many years ahead to forecast? [default {default}]: ").strip()
        if not raw:
            return default
        val = int(raw)
        if val <= 0:
            return default
        return min(val, 50)
    except Exception:
        return default


def main() -> None:
    df = load_water_use_data(DATA_PATH)
    basic_eda(df)
    plot_trends(df)
    plot_households_vs_activities(df)
    plot_fresh_vs_salt_surface(df)

    horizon = prompt_horizon()
    forecast_df = forecast_total_economy(df, horizon_years=horizon)
    if not forecast_df.empty:
        ensure_plots_dir()
        forecast_csv = PLOTS_DIR / "forecast_total_economy.csv"
        forecast_df.to_csv(forecast_csv, index=False)
        print(f"\nForecast saved to {forecast_csv}")
        print(forecast_df.head())
        plot_forecast_total_economy(df, forecast_df)
    else:
        print("\nNot enough data to build a forecast.")

    if OECD_PATH.exists():
        print("\nOECD file found, merging on year")
        oecd_df = load_oecd_data(OECD_PATH)
        water_total = df[df["sector"] == "Total Dutch economy"]
        merged = merge_with_oecd(water_total, oecd_df)
        print("Merged shape:", merged.shape)
        print("Merged columns:", list(merged.columns))
    else:
        print("\nNo OECD file found at", OECD_PATH, "- skipping merge")


if __name__ == "__main__":
    main()
