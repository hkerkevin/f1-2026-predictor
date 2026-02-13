"""
F1 2026 Predictor — Visualizations
Generates plots from Monte Carlo simulation results.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from config import LINEUP_2026

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor": "#181c25",
    "axes.edgecolor": "#2e3440",
    "axes.labelcolor": "#d8dee9",
    "text.color": "#d8dee9",
    "xtick.color": "#d8dee9",
    "ytick.color": "#d8dee9",
    "grid.color": "#2e3440",
    "font.family": "sans-serif",
    "font.size": 11,
})

# F1 team colours (2026 approximations)
TEAM_COLOURS = {
    "McLaren": "#FF8000",
    "Ferrari": "#DC0000",
    "Red Bull Racing": "#3671C6",
    "Mercedes": "#27F4D2",
    "Aston Martin": "#229971",
    "Alpine": "#FF87BC",
    "Haas F1 Team": "#B6BABD",
    "Racing Bulls": "#6692FF",
    "Williams": "#64C4FF",
    "Audi": "#C0C0C0",
    "Cadillac": "#FFD700",
}


def _driver_colour(driver):
    """Return team colour for a driver."""
    for team, drivers in LINEUP_2026.items():
        if driver in drivers:
            return TEAM_COLOURS.get(team, "#888888")
    return "#888888"


def _save(fig, name, out_dir):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── 1. WDC Championship Probability ───────────────────────────────────────
def plot_wdc_championship_probability(sim_results, out_dir):
    n = sim_results["n_simulations"]
    data = sorted(sim_results["champion_counts"].items(), key=lambda x: -x[1])
    # Only show drivers with > 0 probability
    data = [(d, c) for d, c in data if c > 0]

    drivers = [d for d, _ in data]
    probs = [c / n * 100 for _, c in data]
    colours = [_driver_colour(d) for d in drivers]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(drivers[::-1], probs[::-1], color=colours[::-1], edgecolor="none", height=0.65)
    for bar, p in zip(bars, probs[::-1]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{p:.1f}%", va="center", fontsize=10, color="#d8dee9")
    ax.set_xlabel("Championship Probability (%)")
    ax.set_title("2026 Drivers' Championship — Win Probability", fontsize=14, fontweight="bold")
    ax.set_xlim(0, max(probs) * 1.25)
    ax.grid(axis="x", alpha=0.3)
    _save(fig, "wdc_championship_probability.png", out_dir)


# ── 2. WCC Championship Probability ───────────────────────────────────────
def plot_wcc_championship_probability(sim_results, out_dir):
    n = sim_results["n_simulations"]
    data = sorted(sim_results["constructor_champion_counts"].items(), key=lambda x: -x[1])
    data = [(t, c) for t, c in data if c > 0]

    teams = [t for t, _ in data]
    probs = [c / n * 100 for _, c in data]
    colours = [TEAM_COLOURS.get(t, "#888") for t in teams]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(teams[::-1], probs[::-1], color=colours[::-1], edgecolor="none", height=0.6)
    for bar, p in zip(bars, probs[::-1]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{p:.1f}%", va="center", fontsize=10, color="#d8dee9")
    ax.set_xlabel("Championship Probability (%)")
    ax.set_title("2026 Constructors' Championship — Win Probability", fontsize=14, fontweight="bold")
    ax.set_xlim(0, max(probs) * 1.25)
    ax.grid(axis="x", alpha=0.3)
    _save(fig, "wcc_championship_probability.png", out_dir)


# ── 3. Points Distribution (Top 8 Drivers) ────────────────────────────────
def plot_wdc_points_distribution(sim_results, out_dir):
    avg_pts = sorted(
        [(d, np.mean(pts)) for d, pts in sim_results["wdc_points"].items()],
        key=lambda x: -x[1],
    )
    top_drivers = [d for d, _ in avg_pts[:8]]

    fig, ax = plt.subplots(figsize=(12, 6))
    box_data = [sim_results["wdc_points"][d] for d in top_drivers]
    colours = [_driver_colour(d) for d in top_drivers]

    vp = ax.violinplot(box_data, positions=range(len(top_drivers)), showmedians=True,
                        showextrema=False)
    for i, body in enumerate(vp["bodies"]):
        body.set_facecolor(colours[i])
        body.set_alpha(0.7)
    vp["cmedians"].set_color("#ffffff")

    # Overlay median text
    for i, d in enumerate(top_drivers):
        med = np.median(sim_results["wdc_points"][d])
        ax.text(i, med + 12, f"{med:.0f}", ha="center", fontsize=9, color="#ffffff")

    ax.set_xticks(range(len(top_drivers)))
    ax.set_xticklabels(top_drivers, rotation=25, ha="right")
    ax.set_ylabel("Season Points")
    ax.set_title("2026 WDC Points Distribution (5,000 simulations)", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "wdc_points_distribution.png", out_dir)


# ── 4. WCC Points Distribution ────────────────────────────────────────────
def plot_wcc_points_distribution(sim_results, out_dir):
    avg_pts = sorted(
        [(t, np.mean(pts)) for t, pts in sim_results["wcc_points"].items()],
        key=lambda x: -x[1],
    )
    teams = [t for t, _ in avg_pts]

    fig, ax = plt.subplots(figsize=(12, 6))
    box_data = [sim_results["wcc_points"][t] for t in teams]
    colours = [TEAM_COLOURS.get(t, "#888") for t in teams]

    vp = ax.violinplot(box_data, positions=range(len(teams)), showmedians=True,
                        showextrema=False)
    for i, body in enumerate(vp["bodies"]):
        body.set_facecolor(colours[i])
        body.set_alpha(0.7)
    vp["cmedians"].set_color("#ffffff")

    for i, t in enumerate(teams):
        med = np.median(sim_results["wcc_points"][t])
        ax.text(i, med + 15, f"{med:.0f}", ha="center", fontsize=9, color="#ffffff")

    ax.set_xticks(range(len(teams)))
    ax.set_xticklabels(teams, rotation=30, ha="right")
    ax.set_ylabel("Season Points")
    ax.set_title("2026 WCC Points Distribution (5,000 simulations)", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "wcc_points_distribution.png", out_dir)


# ── 5. Average Wins & Poles ───────────────────────────────────────────────
def plot_wins_and_poles(sim_results, out_dir):
    avg_wins = sorted(
        [(d, np.mean(w)) for d, w in sim_results["wdc_wins"].items()],
        key=lambda x: -x[1],
    )
    avg_poles = sorted(
        [(d, np.mean(p)) for d, p in sim_results["wdc_poles"].items()],
        key=lambda x: -x[1],
    )

    # Top 8 for each
    top_win_drivers = [d for d, _ in avg_wins[:8]]
    top_pole_drivers = [d for d, _ in avg_poles[:8]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Wins
    wins_vals = [dict(avg_wins)[d] for d in top_win_drivers]
    colours_w = [_driver_colour(d) for d in top_win_drivers]
    bars1 = ax1.barh(top_win_drivers[::-1], wins_vals[::-1], color=colours_w[::-1],
                      edgecolor="none", height=0.6)
    for bar, v in zip(bars1, wins_vals[::-1]):
        ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                 f"{v:.1f}", va="center", fontsize=10, color="#d8dee9")
    ax1.set_xlabel("Average Race Wins")
    ax1.set_title("Most Race Wins", fontsize=13, fontweight="bold")
    ax1.set_xlim(0, max(wins_vals) * 1.3)
    ax1.grid(axis="x", alpha=0.3)

    # Poles
    poles_vals = [dict(avg_poles)[d] for d in top_pole_drivers]
    colours_p = [_driver_colour(d) for d in top_pole_drivers]
    bars2 = ax2.barh(top_pole_drivers[::-1], poles_vals[::-1], color=colours_p[::-1],
                      edgecolor="none", height=0.6)
    for bar, v in zip(bars2, poles_vals[::-1]):
        ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                 f"{v:.1f}", va="center", fontsize=10, color="#d8dee9")
    ax2.set_xlabel("Average Pole Positions")
    ax2.set_title("Most Pole Positions", fontsize=13, fontweight="bold")
    ax2.set_xlim(0, max(poles_vals) * 1.3)
    ax2.grid(axis="x", alpha=0.3)

    fig.suptitle("2026 Season — Wins & Poles", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "wins_and_poles.png", out_dir)


# ── 6. First Race Winner Probabilities ─────────────────────────────────────
def plot_first_race_winner(sim_results, out_dir):
    n = sim_results["n_simulations"]
    data = sorted(sim_results["first_race_wins"].items(), key=lambda x: -x[1])
    data = [(d, c) for d, c in data if c > 0][:8]

    drivers = [d for d, _ in data]
    probs = [c / n * 100 for _, c in data]
    colours = [_driver_colour(d) for d in drivers]

    fig, ax = plt.subplots(figsize=(8, 5))
    wedges, texts, autotexts = ax.pie(
        probs, labels=drivers, colors=colours, autopct="%1.1f%%",
        startangle=140, pctdistance=0.82,
        wedgeprops=dict(width=0.45, edgecolor="#0f1117", linewidth=1.5),
        textprops={"color": "#d8dee9", "fontsize": 10},
    )
    for t in autotexts:
        t.set_color("#ffffff")
        t.set_fontsize(9)
    # Center label
    ax.text(0, 0, "AUS\n2026", ha="center", va="center",
            fontsize=13, fontweight="bold", color="#d8dee9")
    ax.set_title("First Race Winner — Australia 2026", fontsize=14, fontweight="bold")
    _save(fig, "first_race_winner.png", out_dir)


# ── 7. Teammate Head-to-Head (WDC Points) ─────────────────────────────────
def plot_teammate_comparison(sim_results, out_dir):
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    axes = axes.flatten()

    teams = list(LINEUP_2026.items())
    for i, (team, drivers) in enumerate(teams):
        ax = axes[i]
        d1, d2 = drivers
        pts1 = sim_results["wdc_points"].get(d1, [0])
        pts2 = sim_results["wdc_points"].get(d2, [0])

        colour = TEAM_COLOURS.get(team, "#888")

        ax.hist(pts1, bins=30, alpha=0.65, color=colour, label=d1.split()[-1])
        ax.hist(pts2, bins=30, alpha=0.45, color="#ffffff", edgecolor=colour,
                linewidth=0.8, label=d2.split()[-1])
        ax.set_title(team, fontsize=10, fontweight="bold", color=colour)
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xlabel("Points", fontsize=8)
        ax.tick_params(labelsize=7)

    # Hide extra subplot
    if len(teams) < len(axes):
        for j in range(len(teams), len(axes)):
            axes[j].set_visible(False)

    fig.suptitle("Teammate Head-to-Head — Points Distribution",
                 fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "teammate_comparison.png", out_dir)


# ── 8. WDC Top-3 Position Probabilities (Heatmap) ─────────────────────────
def plot_wdc_top3_heatmap(sim_results, out_dir):
    n = sim_results["n_simulations"]
    avg_pts = sorted(
        [(d, np.mean(pts)) for d, pts in sim_results["wdc_points"].items()],
        key=lambda x: -x[1],
    )
    top_drivers = [d for d, _ in avg_pts[:8]]

    matrix = []
    for d in top_drivers:
        row = [sim_results["wdc_top3_counts"][d].get(pos, 0) / n * 100 for pos in [1, 2, 3]]
        matrix.append(row)

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["P1 (Champion)", "P2", "P3"])
    ax.set_yticks(range(len(top_drivers)))
    ax.set_yticklabels(top_drivers)

    # Annotate cells
    for i in range(len(top_drivers)):
        for j in range(3):
            val = matrix[i, j]
            text_colour = "#000000" if val > 20 else "#d8dee9"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                    fontsize=10, color=text_colour, fontweight="bold")

    ax.set_title("WDC Top-3 Finish Probability", fontsize=14, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="Probability (%)")
    cbar.ax.yaxis.label.set_color("#d8dee9")
    cbar.ax.tick_params(colors="#d8dee9")
    _save(fig, "wdc_top3_heatmap.png", out_dir)


# ── Master function ────────────────────────────────────────────────────────
def generate_all_plots(sim_results, out_dir):
    """Generate all visualizations and save to out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nGenerating plots in {out_dir}/...")

    plot_wdc_championship_probability(sim_results, out_dir)
    plot_wcc_championship_probability(sim_results, out_dir)
    plot_wdc_points_distribution(sim_results, out_dir)
    plot_wcc_points_distribution(sim_results, out_dir)
    plot_wins_and_poles(sim_results, out_dir)
    plot_first_race_winner(sim_results, out_dir)
    plot_teammate_comparison(sim_results, out_dir)
    plot_wdc_top3_heatmap(sim_results, out_dir)

    print(f"  All plots saved to {out_dir}/")
