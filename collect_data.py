"""
F1 2026 Predictor — Data Collection via FastF1
Collects qualifying and race results for historical seasons.
"""

import fastf1
import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

# Use FastF1 default cache location
fastf1.Cache.enable_cache(os.path.join(os.path.dirname(__file__), "f1_cache"))


def get_race_results(year):
    """Get all race results for a given year."""
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    # Filter to only conventional events (races)
    schedule = schedule[schedule["EventFormat"].isin(
        ["conventional", "sprint_shootout", "sprint_qualifying", "sprint"]
    )]

    all_results = []

    for _, event in schedule.iterrows():
        event_name = event["EventName"]
        round_num = event["RoundNumber"]

        try:
            # Load race session
            session = fastf1.get_session(year, round_num, "R")
            session.load(telemetry=False, laps=False, weather=False)
            results = session.results

            if results is None or results.empty:
                continue

            race_df = pd.DataFrame({
                "year": year,
                "round": round_num,
                "event_name": event_name,
                "driver_code": results["Abbreviation"],
                "driver_name": results["FullName"],
                "team": results["TeamName"],
                "grid_position": pd.to_numeric(results["GridPosition"], errors="coerce"),
                "finish_position": pd.to_numeric(results["Position"], errors="coerce"),
                "points": pd.to_numeric(results["Points"], errors="coerce"),
                "status": results["Status"],
                "session_type": "Race",
            })
            all_results.append(race_df)
            print(f"  [R] {year} {event_name}: {len(race_df)} drivers")

        except Exception as e:
            print(f"  [R] {year} {event_name}: FAILED — {e}")

    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()


def get_qualifying_results(year):
    """Get all qualifying results for a given year."""
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    schedule = schedule[schedule["EventFormat"].isin(
        ["conventional", "sprint_shootout", "sprint_qualifying", "sprint"]
    )]

    all_results = []

    for _, event in schedule.iterrows():
        event_name = event["EventName"]
        round_num = event["RoundNumber"]

        try:
            session = fastf1.get_session(year, round_num, "Q")
            session.load(telemetry=False, laps=False, weather=False)
            results = session.results

            if results is None or results.empty:
                continue

            quali_df = pd.DataFrame({
                "year": year,
                "round": round_num,
                "event_name": event_name,
                "driver_code": results["Abbreviation"],
                "driver_name": results["FullName"],
                "team": results["TeamName"],
                "quali_position": pd.to_numeric(results["Position"], errors="coerce"),
                "q1_time": results["Q1"].dt.total_seconds() if "Q1" in results else np.nan,
                "q2_time": results["Q2"].dt.total_seconds() if "Q2" in results else np.nan,
                "q3_time": results["Q3"].dt.total_seconds() if "Q3" in results else np.nan,
                "session_type": "Qualifying",
            })
            all_results.append(quali_df)
            print(f"  [Q] {year} {event_name}: {len(quali_df)} drivers")

        except Exception as e:
            print(f"  [Q] {year} {event_name}: FAILED — {e}")

    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()


def collect_all_data(seasons):
    """Collect race and qualifying data for all specified seasons."""
    all_race = []
    all_quali = []

    for year in seasons:
        print(f"\n{'='*50}")
        print(f"Collecting {year} season data...")
        print(f"{'='*50}")

        for attempt in range(3):
            try:
                race_data = get_race_results(year)
                if not race_data.empty:
                    all_race.append(race_data)
                break
            except Exception as e:
                print(f"  Race collection attempt {attempt+1} failed for {year}: {e}")

        for attempt in range(3):
            try:
                quali_data = get_qualifying_results(year)
                if not quali_data.empty:
                    all_quali.append(quali_data)
                break
            except Exception as e:
                print(f"  Quali collection attempt {attempt+1} failed for {year}: {e}")

    race_df = pd.concat(all_race, ignore_index=True) if all_race else pd.DataFrame()
    quali_df = pd.concat(all_quali, ignore_index=True) if all_quali else pd.DataFrame()

    return race_df, quali_df


if __name__ == "__main__":
    from config import TRAINING_SEASONS

    race_df, quali_df = collect_all_data(TRAINING_SEASONS)
    race_df.to_csv("f1_race_results.csv", index=False)
    quali_df.to_csv("f1_quali_results.csv", index=False)
    print(f"\nCollected {len(race_df)} race entries, {len(quali_df)} qualifying entries")
