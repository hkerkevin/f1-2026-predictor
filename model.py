"""
F1 2026 Predictor — ML Model + Season Simulator
Uses XGBoost to predict finishing positions, then simulates a full season.
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import warnings

from config import (
    LINEUP_2026, F1_POINTS, CALENDAR_2026, ROOKIES_2026, TEAM_NAME_MAP,
)
from features import (
    compute_driver_features, compute_team_features,
    compute_rookie_features, build_training_data,
)

warnings.filterwarnings("ignore")

FEATURE_COLS = [
    "grid_position",
    "rolling_avg_finish",
    "rolling_avg_points",
    "rolling_dnf_rate",
    "rolling_consistency",
    "rolling_win_rate",
    "rolling_podium_rate",
    "rolling_pos_delta",
    "experience",
    "rolling_teammate_delta",
    "team_rolling_avg_finish",
    "team_rolling_avg_points",
    "team_rolling_dnf_rate",
]


def train_position_model(training_df):
    """Train XGBoost model to predict finishing position."""
    # Filter to non-DNF races for position prediction
    df = training_df[training_df["is_dnf"] == 0].copy()
    df = df.dropna(subset=FEATURE_COLS + ["target_position"])

    X = df[FEATURE_COLS]
    y = df["target_position"]

    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
    )

    # Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    scores = cross_val_score(model, X, y, cv=tscv, scoring="neg_mean_absolute_error")
    print(f"  Cross-val MAE: {-scores.mean():.2f} (+/- {scores.std():.2f})")

    # Train on full data
    model.fit(X, y)

    # Feature importance
    importance = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print(f"\n  Feature Importance:")
    for feat, imp in importance.items():
        print(f"    {feat:35s} {imp:.4f}")

    return model


def train_dnf_model(training_df):
    """Train model to predict DNF probability."""
    from xgboost import XGBClassifier

    df = training_df.dropna(subset=FEATURE_COLS).copy()
    X = df[FEATURE_COLS]
    y = df["is_dnf"].astype(int)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X, y)
    return model


def build_2026_driver_features(race_df, quali_df):
    """
    Build feature vectors for each 2026 driver.
    Returns a dict: driver_name → feature dict.
    """
    driver_feats = compute_driver_features(race_df, quali_df)
    team_feats = compute_team_features(race_df, quali_df)
    rookie_feats = compute_rookie_features(driver_feats)

    # Combine
    all_driver_feats = pd.concat([driver_feats, rookie_feats])

    driver_profiles = {}

    for team, drivers in LINEUP_2026.items():
        team_row = team_feats.loc[team] if team in team_feats.index else team_feats.loc["Cadillac"]

        for driver in drivers:
            if driver not in all_driver_feats.index:
                print(f"  WARNING: No features for {driver}, using rookie defaults")
                d = rookie_feats.iloc[0].copy()
            else:
                d = all_driver_feats.loc[driver]

            profile = {
                # Driver features become the "rolling" equivalents
                "rolling_avg_finish": d["recent_avg_finish"],
                "rolling_avg_points": d["recent_avg_points"],
                "rolling_dnf_rate": d["dnf_rate"],
                "rolling_consistency": d["consistency"],
                "rolling_win_rate": d["win_rate"],
                "rolling_podium_rate": d["podium_rate"],
                "rolling_pos_delta": d["position_delta"],
                "experience": d["experience"],
                "rolling_teammate_delta": d["avg_race_delta_vs_teammate"],
                # Team features
                "team_rolling_avg_finish": team_row["recent_avg_finish"],
                "team_rolling_avg_points": team_row["recent_points_per_race"],
                "team_rolling_dnf_rate": team_row["team_dnf_rate"],
                # Extra context for qualifying simulation (use recent form)
                "team": team,
                "avg_quali_pos": d["recent_avg_quali_pos"],
                "team_avg_quali": team_row["recent_avg_team_quali"],
                "reg_adaptability": team_row["reg_adaptability"],
                "trajectory": team_row["trajectory"],
            }
            driver_profiles[driver] = profile

    return driver_profiles


def simulate_qualifying(driver_profiles, rng):
    """
    Simulate qualifying for one race.
    Uses driver quali skill + team car performance + randomness.
    """
    quali_scores = {}

    for driver, profile in driver_profiles.items():
        # Base qualifying performance = blend of driver quali ability + team car performance
        base = 0.4 * profile["avg_quali_pos"] + 0.6 * profile["team_avg_quali"]

        # Regulation change: teams with better adaptability get a boost
        reg_adj = -profile["reg_adaptability"] * 0.3
        trajectory_adj = -profile["trajectory"] * 0.5

        # Random variance (qualifying is high variance)
        noise = rng.normal(0, 1.8)

        quali_scores[driver] = base + reg_adj + trajectory_adj + noise

    # Sort by score (lower = better position)
    sorted_drivers = sorted(quali_scores.items(), key=lambda x: x[1])
    quali_result = {driver: pos + 1 for pos, (driver, _) in enumerate(sorted_drivers)}

    return quali_result


def simulate_race(position_model, dnf_model, driver_profiles, quali_result, rng,
                   _feature_template=None):
    """
    Simulate a single race using the trained model.
    Batches all driver predictions into single model calls for performance.
    """
    drivers = list(driver_profiles.keys())
    n = len(drivers)

    # Build feature matrix for all drivers at once
    rows = []
    for driver in drivers:
        profile = driver_profiles[driver]
        features = {col: profile.get(col, 0) for col in FEATURE_COLS}
        features["grid_position"] = quali_result[driver]
        rows.append(features)

    X = pd.DataFrame(rows, columns=FEATURE_COLS)

    # Batch predict DNF probabilities
    dnf_probs = dnf_model.predict_proba(X)[:, 1]
    is_dnf = rng.random(n) < dnf_probs

    # Batch predict finishing positions
    pred_positions = position_model.predict(X)
    noise = rng.normal(0, 1.2, n)
    pred_positions = np.maximum(1, pred_positions + noise)

    # Build results
    results = []
    for i, driver in enumerate(drivers):
        if is_dnf[i]:
            results.append({
                "driver": driver,
                "team": driver_profiles[driver]["team"],
                "grid": quali_result[driver],
                "predicted_pos": 99,
                "is_dnf": True,
                "points": 0,
            })
        else:
            results.append({
                "driver": driver,
                "team": driver_profiles[driver]["team"],
                "grid": quali_result[driver],
                "predicted_pos": pred_positions[i],
                "is_dnf": False,
                "points": 0,
            })

    # Sort finishers by predicted position, DNFs at back
    results.sort(key=lambda x: (x["is_dnf"], x["predicted_pos"]))

    # Assign actual positions and points
    for i, r in enumerate(results):
        if not r["is_dnf"]:
            actual_pos = i + 1
            r["finish_position"] = actual_pos
            r["points"] = F1_POINTS.get(actual_pos, 0)
        else:
            r["finish_position"] = "DNF"
            r["points"] = 0

    return results


def simulate_season(position_model, dnf_model, driver_profiles, n_simulations=2000):
    """
    Monte Carlo simulation of the full 2026 season.
    Runs many simulations to get probabilistic predictions.
    """
    n_races = len(CALENDAR_2026)
    drivers = list(driver_profiles.keys())

    # Accumulators
    wdc_points = {d: [] for d in drivers}
    wdc_wins = {d: [] for d in drivers}
    wdc_poles = {d: [] for d in drivers}
    wcc_points = {t: [] for t in LINEUP_2026.keys()}
    first_race_wins = {d: 0 for d in drivers}
    champion_counts = {d: 0 for d in drivers}
    constructor_champion_counts = {t: 0 for t in LINEUP_2026.keys()}

    # Track top-3 finishes for WDC/WCC
    wdc_top3_counts = {d: {1: 0, 2: 0, 3: 0} for d in drivers}
    wcc_top3_counts = {t: {1: 0, 2: 0, 3: 0} for t in LINEUP_2026.keys()}

    for sim in range(n_simulations):
        if sim % 200 == 0:
            print(f"  Simulation {sim}/{n_simulations}...")
        rng = np.random.default_rng(seed=sim)

        season_driver_points = {d: 0 for d in drivers}
        season_driver_wins = {d: 0 for d in drivers}
        season_driver_poles = {d: 0 for d in drivers}
        season_team_points = {t: 0 for t in LINEUP_2026.keys()}

        for race_idx, race_name in enumerate(CALENDAR_2026):
            # Simulate qualifying
            quali = simulate_qualifying(driver_profiles, rng)

            # Track poles
            pole_sitter = min(quali, key=quali.get)
            season_driver_poles[pole_sitter] += 1

            # Simulate race
            race_results = simulate_race(
                position_model, dnf_model, driver_profiles, quali, rng
            )

            # Accumulate results
            for r in race_results:
                season_driver_points[r["driver"]] += r["points"]
                season_team_points[r["team"]] += r["points"]
                if r.get("finish_position") == 1:
                    season_driver_wins[r["driver"]] += 1

            # First race winner
            if race_idx == 0:
                winner = [r for r in race_results if r.get("finish_position") == 1]
                if winner:
                    first_race_wins[winner[0]["driver"]] += 1

        # Record season totals
        for d in drivers:
            wdc_points[d].append(season_driver_points[d])
            wdc_wins[d].append(season_driver_wins[d])
            wdc_poles[d].append(season_driver_poles[d])

        for t in LINEUP_2026.keys():
            wcc_points[t].append(season_team_points[t])

        # WDC champion
        wdc_sorted = sorted(season_driver_points.items(), key=lambda x: -x[1])
        champion_counts[wdc_sorted[0][0]] += 1
        for rank_idx in range(3):
            wdc_top3_counts[wdc_sorted[rank_idx][0]][rank_idx + 1] += 1

        # WCC champion
        wcc_sorted = sorted(season_team_points.items(), key=lambda x: -x[1])
        constructor_champion_counts[wcc_sorted[0][0]] += 1
        for rank_idx in range(3):
            wcc_top3_counts[wcc_sorted[rank_idx][0]][rank_idx + 1] += 1

    return {
        "wdc_points": wdc_points,
        "wdc_wins": wdc_wins,
        "wdc_poles": wdc_poles,
        "wcc_points": wcc_points,
        "first_race_wins": first_race_wins,
        "champion_counts": champion_counts,
        "constructor_champion_counts": constructor_champion_counts,
        "wdc_top3_counts": wdc_top3_counts,
        "wcc_top3_counts": wcc_top3_counts,
        "n_simulations": n_simulations,
    }


def generate_predictions(sim_results):
    """
    Generate final predictions from simulation results.
    Outputs the optimal picks for each scoring category.
    """
    n = sim_results["n_simulations"]

    print("\n" + "=" * 70)
    print("  F1 2026 SEASON PREDICTIONS")
    print("  Based on {:,} Monte Carlo simulations".format(n))
    print("=" * 70)

    # --- WDC Champion (25 pts) ---
    print("\n  DRIVERS' CHAMPIONSHIP (WDC)")
    print("  " + "-" * 50)
    wdc_sorted = sorted(sim_results["champion_counts"].items(), key=lambda x: -x[1])
    for driver, count in wdc_sorted[:10]:
        avg_pts = np.mean(sim_results["wdc_points"][driver])
        print(f"    {driver:25s}  Champion: {count/n*100:5.1f}%  Avg Pts: {avg_pts:.0f}")

    predicted_wdc_champion = wdc_sorted[0][0]

    # --- WDC Top 3 (10 pts each) ---
    print(f"\n  Predicted WDC Top 3:")
    wdc_top3 = []
    for pos in [1, 2, 3]:
        best = max(sim_results["wdc_top3_counts"].items(), key=lambda x: x[1][pos])
        # Use expected position ranking instead
    # Better approach: rank by average points
    avg_points_ranked = sorted(
        [(d, np.mean(pts)) for d, pts in sim_results["wdc_points"].items()],
        key=lambda x: -x[1]
    )
    for i, (driver, avg_pts) in enumerate(avg_points_ranked[:3]):
        prob = sim_results["wdc_top3_counts"][driver][i + 1] / n * 100
        print(f"    P{i+1}: {driver:25s}  (Avg {avg_pts:.0f} pts, P{i+1} prob: {prob:.1f}%)")
        wdc_top3.append(driver)

    # --- WCC Champion (20 pts) ---
    print("\n  CONSTRUCTORS' CHAMPIONSHIP (WCC)")
    print("  " + "-" * 50)
    wcc_sorted = sorted(sim_results["constructor_champion_counts"].items(), key=lambda x: -x[1])
    for team, count in wcc_sorted:
        avg_pts = np.mean(sim_results["wcc_points"][team])
        print(f"    {team:25s}  Champion: {count/n*100:5.1f}%  Avg Pts: {avg_pts:.0f}")

    predicted_wcc_champion = wcc_sorted[0][0]

    # --- WCC Top 3 ---
    print(f"\n  Predicted WCC Top 3:")
    wcc_avg_ranked = sorted(
        [(t, np.mean(pts)) for t, pts in sim_results["wcc_points"].items()],
        key=lambda x: -x[1]
    )
    wcc_top3 = []
    for i, (team, avg_pts) in enumerate(wcc_avg_ranked[:3]):
        prob = sim_results["wcc_top3_counts"][team][i + 1] / n * 100
        print(f"    P{i+1}: {team:25s}  (Avg {avg_pts:.0f} pts, P{i+1} prob: {prob:.1f}%)")
        wcc_top3.append(team)

    # --- Most Wins (10 pts) ---
    print("\n  MOST RACE WINS")
    print("  " + "-" * 50)
    avg_wins = sorted(
        [(d, np.mean(w)) for d, w in sim_results["wdc_wins"].items()],
        key=lambda x: -x[1]
    )
    for driver, wins in avg_wins[:5]:
        print(f"    {driver:25s}  Avg Wins: {wins:.1f}")
    predicted_most_wins = avg_wins[0][0]

    # --- Most Poles (10 pts) ---
    print("\n  MOST POLE POSITIONS")
    print("  " + "-" * 50)
    avg_poles = sorted(
        [(d, np.mean(p)) for d, p in sim_results["wdc_poles"].items()],
        key=lambda x: -x[1]
    )
    for driver, poles in avg_poles[:5]:
        print(f"    {driver:25s}  Avg Poles: {poles:.1f}")
    predicted_most_poles = avg_poles[0][0]

    # --- First Race Winner (5 pts) ---
    print("\n  FIRST RACE WINNER (Australia)")
    print("  " + "-" * 50)
    first_sorted = sorted(sim_results["first_race_wins"].items(), key=lambda x: -x[1])
    for driver, count in first_sorted[:5]:
        print(f"    {driver:25s}  Probability: {count/n*100:.1f}%")
    predicted_first_winner = first_sorted[0][0]

    # --- FINAL SUBMISSION ---
    print("\n" + "=" * 70)
    print("  OPTIMAL PREDICTION PICKS")
    print("=" * 70)
    print(f"""
    WDC Champion (25 pts):        {predicted_wdc_champion}
    WDC P2 (10 pts):              {wdc_top3[1]}
    WDC P3 (10 pts):              {wdc_top3[2]}

    WCC Champion (20 pts):        {predicted_wcc_champion}
    WCC P2 (10 pts):              {wcc_top3[1]}
    WCC P3 (10 pts):              {wcc_top3[2]}

    Most Wins (10 pts):           {predicted_most_wins}
    Most Poles (10 pts):          {predicted_most_poles}
    First Race Winner (5 pts):    {predicted_first_winner}
    """)
    print("=" * 70)

    return {
        "wdc_champion": predicted_wdc_champion,
        "wdc_top3": wdc_top3,
        "wcc_champion": predicted_wcc_champion,
        "wcc_top3": wcc_top3,
        "most_wins": predicted_most_wins,
        "most_poles": predicted_most_poles,
        "first_race_winner": predicted_first_winner,
    }
