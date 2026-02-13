"""
F1 2026 Season Predictor — Main Pipeline
=========================================
Uses FastF1 historical data + XGBoost + Monte Carlo simulation
to predict the 2026 F1 season outcomes.

Prediction targets:
- WDC Champion + Top 3
- WCC Champion + Top 3
- Most Race Wins
- Most Pole Positions
- First Race Winner
"""

import os
import sys
import pandas as pd
import numpy as np

# Ensure imports work
sys.path.insert(0, os.path.dirname(__file__))

from config import TRAINING_SEASONS, LINEUP_2026
from collect_data import collect_all_data
from features import build_training_data, compute_driver_features, compute_team_features
from model import (
    train_position_model,
    train_dnf_model,
    build_2026_driver_features,
    simulate_season,
    generate_predictions,
    FEATURE_COLS,
)
from visualizations import generate_all_plots


def main():
    data_dir = os.path.dirname(__file__)
    race_csv = os.path.join(data_dir, "f1_race_results.csv")
    quali_csv = os.path.join(data_dir, "f1_quali_results.csv")

    # ── Step 1: Collect Data ──────────────────────────────────────────────
    if os.path.exists(race_csv) and os.path.exists(quali_csv):
        print("Found cached data files, loading...")
        race_df = pd.read_csv(race_csv)
        quali_df = pd.read_csv(quali_csv)
        print(f"  Race entries: {len(race_df)}")
        print(f"  Qualifying entries: {len(quali_df)}")
    else:
        print("Collecting historical F1 data via FastF1...")
        print("(This downloads session data — first run may be slow)\n")
        race_df, quali_df = collect_all_data(TRAINING_SEASONS)

        race_df.to_csv(race_csv, index=False)
        quali_df.to_csv(quali_csv, index=False)
        print(f"\nSaved {len(race_df)} race entries, {len(quali_df)} qualifying entries")

    # ── Step 2: Build Training Data ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("Building training dataset with rolling features...")
    print("=" * 60)
    training_df = build_training_data(race_df, quali_df)
    print(f"  Training samples: {len(training_df)}")
    print(f"  Unique drivers: {training_df['driver'].nunique()}")
    print(f"  Feature columns: {len(FEATURE_COLS)}")

    # ── Step 3: Train Models ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Training XGBoost position prediction model...")
    print("=" * 60)
    position_model = train_position_model(training_df)

    print("\n" + "=" * 60)
    print("Training DNF probability model...")
    print("=" * 60)
    dnf_model = train_dnf_model(training_df)

    # ── Step 4: Build 2026 Driver Profiles ────────────────────────────────
    print("\n" + "=" * 60)
    print("Building 2026 driver profiles...")
    print("=" * 60)
    driver_profiles = build_2026_driver_features(race_df, quali_df)

    print("\n  2026 Grid:")
    for team, drivers in LINEUP_2026.items():
        for d in drivers:
            prof = driver_profiles[d]
            print(f"    {d:25s} ({team:20s})  "
                  f"AvgFinish={prof['rolling_avg_finish']:.1f}  "
                  f"TeamCar={prof['team_rolling_avg_finish']:.1f}")

    # ── Step 5: Monte Carlo Season Simulation ─────────────────────────────
    print("\n" + "=" * 60)
    print("Running Monte Carlo season simulation (2,000 seasons)...")
    print("=" * 60)
    sim_results = simulate_season(position_model, dnf_model, driver_profiles, n_simulations=5000)

    # ── Step 6: Generate Predictions ──────────────────────────────────────
    predictions = generate_predictions(sim_results)

    # ── Step 7: Generate Visualizations ─────────────────────────────────
    plot_dir = os.path.join(data_dir, "plots")
    generate_all_plots(sim_results, plot_dir)

    return predictions


if __name__ == "__main__":
    predictions = main()
