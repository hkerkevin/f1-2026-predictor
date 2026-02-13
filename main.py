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

import argparse
import os
import sys
import json
import glob
from datetime import datetime
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


def main(generate_plots=False):
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
    print("Running Monte Carlo season simulation (5,000 seasons)...")
    print("=" * 60)
    sim_results = simulate_season(position_model, dnf_model, driver_profiles, n_simulations=5000)

    # ── Step 6: Generate Predictions ──────────────────────────────────────
    predictions = generate_predictions(sim_results)

    # ── Step 7: Generate Visualizations (optional) ──────────────────────
    if generate_plots:
        from visualizations import generate_all_plots
        plot_dir = os.path.join(data_dir, "plots")
        generate_all_plots(sim_results, plot_dir)
    else:
        print("\n  Skipping plot generation (use --plots to regenerate)")

    # ── Step 8: Save Versioned Predictions ──────────────────────────────
    save_versioned_predictions(data_dir, predictions, sim_results, FEATURE_COLS)

    return predictions


def save_versioned_predictions(data_dir, predictions, sim_results, feature_cols):
    """Save predictions to a versioned JSON file and compare with previous runs."""
    pred_dir = os.path.join(data_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    # Find next version number
    existing = sorted(glob.glob(os.path.join(pred_dir, "v*.json")))
    version = len(existing) + 1

    # Build detailed results
    n = sim_results["n_simulations"]
    avg_driver_pts = {d: float(np.mean(pts)) for d, pts in sim_results["wdc_points"].items()}
    avg_team_pts = {t: float(np.mean(pts)) for t, pts in sim_results["wcc_points"].items()}
    champion_probs = {d: round(c / n * 100, 1) for d, c in sim_results["champion_counts"].items() if c > 0}
    wcc_probs = {t: round(c / n * 100, 1) for t, c in sim_results["constructor_champion_counts"].items() if c > 0}

    record = {
        "version": f"v{version}",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_simulations": n,
        "features": feature_cols,
        "predictions": predictions,
        "driver_avg_points": avg_driver_pts,
        "team_avg_points": avg_team_pts,
        "wdc_champion_probabilities": champion_probs,
        "wcc_champion_probabilities": wcc_probs,
    }

    filename = f"v{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(pred_dir, filename)
    with open(filepath, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\n  Saved predictions to {filepath}")

    # Compare with previous version if available
    if existing:
        prev_path = existing[-1]
        with open(prev_path) as f:
            prev = json.load(f)
        print(f"\n  === COMPARISON: {prev['version']} -> v{version} ===")
        print(f"  {'Category':<30s} {'Previous':<25s} {'Current':<25s} {'Changed'}")
        print(f"  {'-'*95}")
        for key in ["wdc_champion", "most_wins", "most_poles", "first_race_winner"]:
            old_val = prev["predictions"].get(key, "?")
            new_val = predictions.get(key, "?")
            changed = " *" if old_val != new_val else ""
            print(f"  {key:<30s} {old_val:<25s} {new_val:<25s}{changed}")
        for i in range(3):
            key = f"wdc_p{i+1}"
            old_val = prev["predictions"].get("wdc_top3", ["?", "?", "?"])[i]
            new_val = predictions.get("wdc_top3", ["?", "?", "?"])[i]
            changed = " *" if old_val != new_val else ""
            print(f"  {key:<30s} {old_val:<25s} {new_val:<25s}{changed}")
        for i in range(3):
            key = f"wcc_p{i+1}"
            old_val = prev["predictions"].get("wcc_top3", ["?", "?", "?"])[i]
            new_val = predictions.get("wcc_top3", ["?", "?", "?"])[i]
            changed = " *" if old_val != new_val else ""
            print(f"  {key:<30s} {old_val:<25s} {new_val:<25s}{changed}")
        print(f"  Features: {len(prev.get('features', []))} -> {len(feature_cols)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F1 2026 Season Predictor")
    parser.add_argument("--plots", action="store_true", help="Regenerate visualization plots")
    args = parser.parse_args()
    predictions = main(generate_plots=args.plots)
