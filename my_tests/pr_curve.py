# plotting
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

# data handling
import numpy as np
import pandas as pd
import glob


GLOBAL_COLORS = {
    "companies": "darkorange",
    "movies": "steelblue",
    "SN": "green",
    "Round1_T2D": "red",
    "Round4_2020": "purple",
    "HardTablesR2": "brown"
}


def load_latest_confidence_file():
    """Loads the most recent confidence_scores CSV from logs."""
    files = glob.glob("my_tests/logs/confidence_scores_*.csv")
    if not files:
        raise FileNotFoundError("No confidence score files found. Run datasets.py first.")
    latest = max(files)
    print(f"Loading: {latest}")
    return pd.read_csv(latest)


def plot_confidence_results(df, datasets, colors, title_suffix="", mode="cell"):
    """Plots PR curve and F1 vs Threshold for selected datasets."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    plt.suptitle(f"ReFinED Confidence Analysis {title_suffix}", fontsize=13, fontweight="bold")

    # loop through datasets and plot
    for ds in datasets:
        subset = df[(df["dataset"] == ds) & (df["mode"] == mode)]
        if subset.empty:
            print(f"No data for {ds} in mode {mode}, skipping.")
            continue

        # get scores and labels
        scores = subset["confidence"].values
        labels = subset["is_correct"].values
        c = colors.get(ds, "black")

        # calculate PR curve and F1 scores
        p, r, t = precision_recall_curve(labels, scores)
        t_full = np.clip(np.append(t, 1.0), 0.01, 1.0)
        f1 = 2 * (p * r) / (p + r + 1e-8)
        pr_auc = auc(r, p)
        best_f1 = np.max(f1)

        # plotting
        axes[0].plot(r, p, color=c, lw=2, label=f"{ds} (AUC={pr_auc:.3f})")
        axes[1].plot(t_full, f1, color=c, lw=2, label=f"{ds} (Peak={best_f1:.2f})")

    for ax in axes:
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
        ax.legend()

    # labels and titles
    axes[0].set_xlabel("Recall"), axes[0].set_ylabel("Precision")
    axes[0].set_title("Precision-Recall Curve")
    axes[1].set_xlabel("Confidence Threshold"), axes[1].set_ylabel("F1 Score")
    axes[1].set_title("F1 Score vs Confidence Threshold")

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # TODO: THIS SCRIPT REQUIRES RUNNING 'datasets.py' FIRST TO GENERATE LOGS
    df = load_latest_confidence_file()

    for mode in ["cell", "row"]:
        # Specialized
        spec_ids = ["companies", "movies", "SN"]
        plot_confidence_results(df, spec_ids, GLOBAL_COLORS, f"- Specialized ({mode})", mode=mode) \
            .savefig(f"my_tests/logs/confidence_specialized_{mode}.png", dpi=150, bbox_inches="tight")

        # SemTab
        semtab_ids = ["Round1_T2D", "Round3_2019", "Round4_2020", "HardTablesR2", "HardTablesR3"]
        plot_confidence_results(df, semtab_ids, GLOBAL_COLORS, f"- SemTab Rounds ({mode})", mode=mode) \
            .savefig(f"my_tests/logs/confidence_semtab_{mode}.png", dpi=150, bbox_inches="tight")

print("Done.")