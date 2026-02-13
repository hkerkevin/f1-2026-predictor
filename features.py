"""
F1 2026 Predictor — Feature Engineering
Builds driver skill ratings, team strength ratings, and circuit features.
"""

import pandas as pd
import numpy as np
from config import TEAM_NAME_MAP, DRIVER_NAME_MAP, F1_POINTS, ROOKIES_2026


def normalize_team_name(team_name):
    """Map any historical team name to its 2026 identity."""
    for team_2026, aliases in TEAM_NAME_MAP.items():
        if team_name in aliases or team_name == team_2026:
            return team_2026
    return team_name


def normalize_driver_name(driver_name):
    """Map any historical driver name variant to canonical 2026 name."""
    for canonical, aliases in DRIVER_NAME_MAP.items():
        if driver_name in aliases or driver_name == canonical:
            return canonical
    return driver_name


def match_driver_by_code(code):
    """Match a driver abbreviation to canonical name."""
    for canonical, aliases in DRIVER_NAME_MAP.items():
        if code in aliases:
            return canonical
    return None


def add_dnf_flag(race_df):
    """Add a binary DNF column."""
    dnf_statuses = ["Retired", "Accident", "Collision", "Engine", "Gearbox",
                    "Hydraulics", "Brakes", "Electrical", "Suspension",
                    "Mechanical", "Puncture", "Power Unit", "Disqualified",
                    "Withdrew", "Excluded"]
    race_df = race_df.copy()
    race_df["is_dnf"] = race_df["status"].apply(
        lambda s: 0 if s in ["Finished", "+1 Lap", "+2 Laps", "+3 Laps"] else 1
    )
    return race_df


def compute_driver_features(race_df, quali_df):
    """
    Compute driver-level features that isolate driver skill from car performance.

    Key features:
    - Teammate qualifying delta (driver skill in qualifying)
    - Teammate race finishing delta (driver skill in race)
    - Position gains/losses from grid (racecraft / first lap ability)
    - Consistency (std dev of finishing positions)
    - DNF rate
    - Points per race
    - Experience (race count)
    - Win rate, podium rate
    """
    race_df = add_dnf_flag(race_df)
    race_df["norm_driver"] = race_df["driver_name"].apply(normalize_driver_name)
    race_df["norm_team"] = race_df["team"].apply(normalize_team_name)

    quali_df = quali_df.copy()
    quali_df["norm_driver"] = quali_df["driver_name"].apply(normalize_driver_name)
    quali_df["norm_team"] = quali_df["team"].apply(normalize_team_name)

    features = {}

    # Get list of all drivers on the 2026 grid (that have F1 history)
    target_drivers = set(DRIVER_NAME_MAP.keys()) - set(ROOKIES_2026)

    for driver in target_drivers:
        d_races = race_df[race_df["norm_driver"] == driver].copy()
        d_quali = quali_df[quali_df["norm_driver"] == driver].copy()

        if d_races.empty:
            continue

        # --- Basic stats ---
        finished_races = d_races[d_races["is_dnf"] == 0]
        total_races = len(d_races)

        avg_finish = finished_races["finish_position"].mean() if not finished_races.empty else 15.0
        avg_grid = d_races["grid_position"].dropna().mean()
        avg_points = d_races["points"].mean()
        win_rate = (finished_races["finish_position"] == 1).sum() / max(total_races, 1)
        podium_rate = (finished_races["finish_position"] <= 3).sum() / max(total_races, 1)
        dnf_rate = d_races["is_dnf"].mean()
        consistency = finished_races["finish_position"].std() if len(finished_races) > 1 else 8.0
        experience = total_races

        # --- Position delta (grid → finish): racecraft ---
        d_races_valid = d_races[(d_races["is_dnf"] == 0) & d_races["grid_position"].notna()]
        if not d_races_valid.empty:
            position_delta = (d_races_valid["grid_position"] - d_races_valid["finish_position"]).mean()
        else:
            position_delta = 0.0

        # --- Teammate qualifying delta ---
        # For each race, compare this driver's quali to teammate
        quali_deltas = []
        for (year, rnd), group in d_quali.groupby(["year", "round"]):
            team = group["norm_team"].iloc[0]
            teammates = quali_df[
                (quali_df["year"] == year) &
                (quali_df["round"] == rnd) &
                (quali_df["norm_team"] == team) &
                (quali_df["norm_driver"] != driver)
            ]
            if not teammates.empty:
                driver_pos = group["quali_position"].iloc[0]
                teammate_pos = teammates["quali_position"].iloc[0]
                if pd.notna(driver_pos) and pd.notna(teammate_pos):
                    quali_deltas.append(teammate_pos - driver_pos)  # Positive = beat teammate

        avg_quali_delta = np.mean(quali_deltas) if quali_deltas else 0.0

        # --- Teammate race delta ---
        race_deltas = []
        for (year, rnd), group in d_races[d_races["is_dnf"] == 0].groupby(["year", "round"]):
            team = group["norm_team"].iloc[0]
            teammates = race_df[
                (race_df["year"] == year) &
                (race_df["round"] == rnd) &
                (race_df["norm_team"] == team) &
                (race_df["norm_driver"] != driver) &
                (race_df["is_dnf"] == 0)
            ]
            if not teammates.empty:
                driver_pos = group["finish_position"].iloc[0]
                teammate_pos = teammates["finish_position"].iloc[0]
                if pd.notna(driver_pos) and pd.notna(teammate_pos):
                    race_deltas.append(teammate_pos - driver_pos)  # Positive = beat teammate

        avg_race_delta = np.mean(race_deltas) if race_deltas else 0.0

        # --- Recent form: weighted toward latest seasons ---
        recent_races = d_races[d_races["year"] >= d_races["year"].max() - 1]
        recent_finished = recent_races[recent_races["is_dnf"] == 0]
        recent_avg_finish = recent_finished["finish_position"].mean() if not recent_finished.empty else avg_finish
        recent_avg_points = recent_races["points"].mean()

        # --- Qualifying pace (average quali position) ---
        avg_quali_pos = d_quali["quali_position"].mean() if not d_quali.empty else 12.0

        # Recent qualifying pace (last 2 seasons — much more relevant for 2026)
        recent_quali = d_quali[d_quali["year"] >= d_quali["year"].max() - 1] if not d_quali.empty else d_quali
        recent_avg_quali_pos = recent_quali["quali_position"].mean() if not recent_quali.empty else avg_quali_pos

        features[driver] = {
            "avg_finish": avg_finish,
            "avg_grid": avg_grid if pd.notna(avg_grid) else 12.0,
            "avg_points": avg_points,
            "win_rate": win_rate,
            "podium_rate": podium_rate,
            "dnf_rate": dnf_rate,
            "consistency": consistency,
            "experience": experience,
            "position_delta": position_delta,
            "avg_quali_delta_vs_teammate": avg_quali_delta,
            "avg_race_delta_vs_teammate": avg_race_delta,
            "recent_avg_finish": recent_avg_finish,
            "recent_avg_points": recent_avg_points,
            "avg_quali_pos": avg_quali_pos,
            "recent_avg_quali_pos": recent_avg_quali_pos,
        }

    return pd.DataFrame(features).T


def compute_team_features(race_df, quali_df):
    """
    Compute team-level features.

    Key features:
    - Average constructor points per race
    - Average car finishing position
    - Reliability (team DNF rate)
    - Development rate (improvement across seasons)
    - Regulation change adaptability (2022 performance vs 2021)
    - Recent form
    """
    race_df = race_df.copy()
    race_df["norm_team"] = race_df["team"].apply(normalize_team_name)
    race_df = add_dnf_flag(race_df)

    quali_df = quali_df.copy()
    quali_df["norm_team"] = quali_df["team"].apply(normalize_team_name)

    features = {}

    for team in TEAM_NAME_MAP.keys():
        t_races = race_df[race_df["norm_team"] == team]

        if t_races.empty:
            # New team (Cadillac) — assign baseline features
            features[team] = _default_team_features()
            continue

        finished = t_races[t_races["is_dnf"] == 0]

        # Points per race (both drivers combined, then averaged per race)
        points_per_race = t_races.groupby(["year", "round"])["points"].sum().mean()

        # Average car finishing position
        avg_car_finish = finished["finish_position"].mean() if not finished.empty else 12.0

        # Team reliability
        team_dnf_rate = t_races["is_dnf"].mean()

        # Average quali position
        t_quali = quali_df[quali_df["norm_team"] == team]
        avg_team_quali = t_quali["quali_position"].mean() if not t_quali.empty else 12.0

        # Recent qualifying pace (last 2 seasons — more relevant for 2026)
        if not t_quali.empty:
            recent_t_quali = t_quali[t_quali["year"] >= t_quali["year"].max() - 1]
            recent_avg_team_quali = recent_t_quali["quali_position"].mean() if not recent_t_quali.empty else avg_team_quali
        else:
            recent_avg_team_quali = avg_team_quali

        # --- Development rate: improvement from first half to second half of recent season ---
        latest_year = t_races["year"].max()
        latest_races = t_races[t_races["year"] == latest_year]
        if len(latest_races) > 4:
            latest_finished = latest_races[latest_races["is_dnf"] == 0]
            mid = len(latest_finished) // 2
            first_half = latest_finished.iloc[:mid]["finish_position"].mean()
            second_half = latest_finished.iloc[mid:]["finish_position"].mean()
            dev_rate = first_half - second_half  # Positive = improved (lower positions = better)
        else:
            dev_rate = 0.0

        # --- Regulation change adaptability ---
        # Compare 2022 (new regs) vs 2021 (old regs) performance
        pre_reg = t_races[t_races["year"] == 2021]
        post_reg = t_races[t_races["year"] == 2022]
        if not pre_reg.empty and not post_reg.empty:
            pre_avg = pre_reg[pre_reg["is_dnf"] == 0]["finish_position"].mean()
            post_avg = post_reg[post_reg["is_dnf"] == 0]["finish_position"].mean()
            reg_adaptability = pre_avg - post_avg  # Positive = improved after reg change
        else:
            reg_adaptability = 0.0

        # --- Recent form (last 2 seasons) ---
        recent = t_races[t_races["year"] >= latest_year - 1]
        recent_finished = recent[recent["is_dnf"] == 0]
        recent_avg_finish = recent_finished["finish_position"].mean() if not recent_finished.empty else avg_car_finish
        recent_points = recent.groupby(["year", "round"])["points"].sum().mean()

        # --- Season-over-season trajectory ---
        yearly_avg = t_races.groupby("year").apply(
            lambda g: g[g["is_dnf"] == 0]["finish_position"].mean()
        )
        if len(yearly_avg) >= 2:
            trajectory = yearly_avg.iloc[-2] - yearly_avg.iloc[-1]  # Positive = improving
        else:
            trajectory = 0.0

        features[team] = {
            "points_per_race": points_per_race,
            "avg_car_finish": avg_car_finish,
            "team_dnf_rate": team_dnf_rate,
            "avg_team_quali": avg_team_quali,
            "recent_avg_team_quali": recent_avg_team_quali,
            "dev_rate": dev_rate,
            "reg_adaptability": reg_adaptability,
            "recent_avg_finish": recent_avg_finish,
            "recent_points_per_race": recent_points,
            "trajectory": trajectory,
        }

    return pd.DataFrame(features).T


def _default_team_features():
    """Default features for a brand-new team (Cadillac)."""
    return {
        "points_per_race": 0.5,
        "avg_car_finish": 16.0,
        "team_dnf_rate": 0.15,
        "avg_team_quali": 17.0,
        "recent_avg_team_quali": 17.0,
        "dev_rate": 0.0,
        "reg_adaptability": 0.0,
        "recent_avg_finish": 16.0,
        "recent_points_per_race": 0.5,
        "trajectory": 0.0,
    }


def compute_rookie_features(driver_features_df):
    """
    Estimate rookie features using proxy data.
    Rookies are assigned slightly below-average features with high uncertainty.
    We use percentile-based estimates: rookies start around P55 of current grid.
    """
    median_feats = driver_features_df.median()

    rookie_features = {}
    for rookie in ROOKIES_2026:
        feats = median_feats.copy()
        # Rookies typically finish lower, have less consistency
        feats["avg_finish"] = median_feats["avg_finish"] + 2.0
        feats["recent_avg_finish"] = median_feats["recent_avg_finish"] + 2.0
        feats["avg_grid"] = median_feats["avg_grid"] + 2.0
        feats["avg_quali_pos"] = median_feats["avg_quali_pos"] + 2.0
        feats["recent_avg_quali_pos"] = median_feats["recent_avg_quali_pos"] + 2.0
        feats["avg_points"] = median_feats["avg_points"] * 0.4
        feats["recent_avg_points"] = median_feats["recent_avg_points"] * 0.4
        feats["win_rate"] = 0.0
        feats["podium_rate"] = 0.01
        feats["consistency"] = median_feats["consistency"] + 2.0
        feats["experience"] = 5  # Minimal
        feats["position_delta"] = 0.0
        feats["avg_quali_delta_vs_teammate"] = -1.0  # Slightly behind teammate
        feats["avg_race_delta_vs_teammate"] = -0.5
        rookie_features[rookie] = feats

    return pd.DataFrame(rookie_features).T


def build_training_data(race_df, quali_df):
    """
    Build training dataset: each row = one driver-race entry with features.
    Target = finishing position.

    Uses vectorized pandas operations for performance.
    Features use only past data (no leakage).
    """
    race_df = race_df.copy()
    race_df = add_dnf_flag(race_df)
    race_df["norm_driver"] = race_df["driver_name"].apply(normalize_driver_name)
    race_df["norm_team"] = race_df["team"].apply(normalize_team_name)

    quali_df = quali_df.copy()
    quali_df["norm_driver"] = quali_df["driver_name"].apply(normalize_driver_name)
    quali_df["norm_team"] = quali_df["team"].apply(normalize_team_name)

    # Merge qualifying data into race data
    merged = race_df.merge(
        quali_df[["year", "round", "norm_driver", "quali_position"]],
        on=["year", "round", "norm_driver"],
        how="left",
    )

    # Sort chronologically and create a race order index
    merged = merged.sort_values(["year", "round", "norm_driver"]).reset_index(drop=True)

    # Compute position delta (grid - finish) for position gain metric
    merged["pos_delta"] = merged["grid_position"] - merged["finish_position"]
    # finish_position for non-DNF only (for rolling stats)
    merged["finish_clean"] = merged["finish_position"].where(merged["is_dnf"] == 0)
    merged["is_win"] = ((merged["finish_position"] == 1) & (merged["is_dnf"] == 0)).astype(float)
    merged["is_podium"] = ((merged["finish_position"] <= 3) & (merged["is_dnf"] == 0)).astype(float)
    merged["pos_delta_clean"] = merged["pos_delta"].where(
        (merged["is_dnf"] == 0) & merged["grid_position"].notna()
    )

    # --- Teammate delta: pre-compute for all races ---
    # For each race, join driver with teammate's finish position
    race_team_groups = merged[merged["is_dnf"] == 0].groupby(
        ["year", "round", "norm_team"]
    )["finish_position"].apply(list).reset_index()

    # For each driver-race, find teammate finish position
    tm_lookup = merged[["year", "round", "norm_team", "norm_driver", "finish_position", "is_dnf"]].copy()
    tm_lookup = tm_lookup.merge(
        tm_lookup, on=["year", "round", "norm_team"], suffixes=("", "_tm")
    )
    tm_lookup = tm_lookup[tm_lookup["norm_driver"] != tm_lookup["norm_driver_tm"]]
    # Only count when both finished
    tm_both = tm_lookup[(tm_lookup["is_dnf"] == 0) & (tm_lookup["is_dnf_tm"] == 0)]
    tm_both["teammate_delta"] = tm_both["finish_position_tm"] - tm_both["finish_position"]
    tm_delta_map = tm_both.groupby(["year", "round", "norm_driver"])["teammate_delta"].mean()

    merged = merged.merge(
        tm_delta_map.reset_index().rename(columns={"teammate_delta": "race_tm_delta"}),
        on=["year", "round", "norm_driver"],
        how="left",
    )

    # --- Rolling features per driver (window=20, shift=1 to avoid leakage) ---
    print("  Computing rolling driver features...")
    merged = merged.sort_values(["norm_driver", "year", "round"]).reset_index(drop=True)

    WIN = 20  # Rolling window size

    def rolling_driver_features(g):
        g = g.sort_values(["year", "round"])
        g["rolling_avg_finish"] = g["finish_clean"].expanding().mean().shift(1)
        g["rolling_avg_points"] = g["points"].rolling(WIN, min_periods=3).mean().shift(1)
        g["rolling_dnf_rate"] = g["is_dnf"].rolling(WIN, min_periods=3).mean().shift(1)
        g["rolling_win_rate"] = g["is_win"].rolling(WIN, min_periods=3).mean().shift(1)
        g["rolling_podium_rate"] = g["is_podium"].rolling(WIN, min_periods=3).mean().shift(1)
        g["rolling_pos_delta"] = g["pos_delta_clean"].rolling(WIN, min_periods=3).mean().shift(1)
        g["rolling_teammate_delta"] = g["race_tm_delta"].rolling(10, min_periods=1).mean().shift(1)
        g["experience"] = range(len(g))
        # Consistency: rolling std of finish positions
        g["rolling_consistency"] = g["finish_clean"].rolling(WIN, min_periods=3).std().shift(1)
        return g

    merged = merged.groupby("norm_driver", group_keys=False).apply(rolling_driver_features)

    # --- Rolling features per team ---
    print("  Computing rolling team features...")
    # Compute team-race-level stats first
    team_race = merged.groupby(["norm_team", "year", "round"]).agg(
        team_points=("points", "sum"),
        team_avg_finish=("finish_clean", "mean"),
        team_dnf_rate=("is_dnf", "mean"),
    ).reset_index().sort_values(["norm_team", "year", "round"])

    def rolling_team_features(g):
        g = g.sort_values(["year", "round"])
        g["team_rolling_avg_finish"] = g["team_avg_finish"].rolling(20, min_periods=2).mean().shift(1)
        g["team_rolling_avg_points"] = g["team_points"].rolling(20, min_periods=2).mean().shift(1)
        g["team_rolling_dnf_rate"] = g["team_dnf_rate"].rolling(20, min_periods=2).mean().shift(1)
        return g

    team_race = team_race.groupby("norm_team", group_keys=False).apply(rolling_team_features)

    merged = merged.merge(
        team_race[["norm_team", "year", "round", "team_rolling_avg_finish",
                    "team_rolling_avg_points", "team_rolling_dnf_rate"]],
        on=["norm_team", "year", "round"],
        how="left",
    )

    # --- Assemble final training data ---
    result = merged[[
        "year", "round", "norm_driver", "norm_team", "grid_position",
        "rolling_avg_finish", "rolling_avg_points", "rolling_dnf_rate",
        "rolling_consistency", "rolling_win_rate", "rolling_podium_rate",
        "rolling_pos_delta", "experience", "rolling_teammate_delta",
        "team_rolling_avg_finish", "team_rolling_avg_points", "team_rolling_dnf_rate",
        "finish_position", "is_dnf",
    ]].copy()

    result = result.rename(columns={
        "norm_driver": "driver",
        "norm_team": "team",
        "finish_position": "target_position",
    })

    # Drop rows without enough history
    result = result.dropna(subset=["rolling_avg_finish", "rolling_avg_points"])
    result = result[result["experience"] >= 3]

    print(f"  Built {len(result)} training samples")
    return result


if __name__ == "__main__":
    race_df = pd.read_csv("f1_race_results.csv")
    quali_df = pd.read_csv("f1_quali_results.csv")

    print("Computing driver features...")
    driver_feats = compute_driver_features(race_df, quali_df)
    print(driver_feats)

    print("\nComputing team features...")
    team_feats = compute_team_features(race_df, quali_df)
    print(team_feats)
