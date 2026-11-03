import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =========================
# 1. Data
# =========================
data = {
    "Model": [
        "HBA-LGBM",
        "TabTransformer-WL",
        "LPBiLSTM+DSKNN+KOA",
        "HMDL-GFNR",
        "STAGE (Ours)",
        "GPGVAE",
        "GraphVAE",
        "Informer",
        "LSTM",
        "MuLVAEK",
        "VAEneu"
    ],
    "LendingClub": [
        10.4106, 3.7127, 1.3495, 6.3650, 8.0921,
        1.36, 24.85, 2.11, 0.90, 1.28, 2.51
    ],
    "GMSC": [
        0.8378, 1.9690, 0.5056, 1.6512, 3.9327,
        0.67, 13.25, 1.06, 0.46, 0.64, 1.25
    ],
    "Chinese Personal Loan": [
        86.1114, 29.7900, 7.9156, 33.1248, 68.4950,
        10.55, 187.78, 14.79, 7.10, 9.84, 19.97
    ],
    "UCI": [
        1.3502, 1.0209, 0.1936, 1.4276, 2.5221,
        0.41, 7.37, 0.65, 0.28, 0.37, 0.76
    ]
}

df = pd.DataFrame(data)

# =========================
# 2. Plotting parameters
# =========================
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

datasets = ["LendingClub", "GMSC", "Chinese Personal Loan", "UCI"]

# 2 rows x 2 columns subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

# Color settings
normal_color = "#4F87A1"
ours_color = "#C0392B"
avg_color = "#666666"

# =========================
# 3. Plot the four datasets separately
# =========================
for ax, dataset in zip(axes, datasets):

    # Sort by training time in ascending order
    plot_df = df[["Model", dataset]].sort_values(by=dataset, ascending=True).reset_index(drop=True)

    models = plot_df["Model"].tolist()
    values = plot_df[dataset].tolist()

    # STAGE (Ours) in red, other models in blue
    colors = [
        ours_color if model == "STAGE (Ours)" else normal_color
        for model in models
    ]

    # Draw horizontal bar chart
    bars = ax.barh(
        models,
        values,
        color=colors,
        edgecolor="white",
        height=0.62
    )

    # Average training time
    avg_time = np.mean(values)
    ax.axvline(
        avg_time,
        color=avg_color,
        linestyle="--",
        linewidth=1.2,
        alpha=0.85
    )

    # Average value text
    max_value = max(values)
    ax.text(
        avg_time + max_value * 0.01,
        len(models) - 0.4,
        f"Avg: {avg_time:.2f}s",
        color=avg_color,
        fontsize=10,
        fontstyle="italic"
    )

    # Show value on the right side of each bar
    for bar, value, model in zip(bars, values, models):
        label_color = ours_color if model == "TSET (Ours)" else "#1F5F7A"

        ax.text(
            bar.get_width() + max_value * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}s",
            va="center",
            ha="left",
            fontsize=9,
            color=label_color,
            fontweight="bold" if model == "TSET (Ours)" else "normal"
        )

    # Title and axes
    ax.set_title(
        f"{dataset}",
        fontsize=14,
        fontweight="bold",
        pad=10
    )
    ax.set_xlabel("Average Training Time (Seconds)", fontsize=11)
    ax.set_ylabel("Model", fontsize=11)

    # Grid lines
    ax.grid(axis="x", linestyle=":", alpha=0.35)

    # Set x-axis display range to leave room for text
    ax.set_xlim(0, max_value * 1.23)

    # Hide top and right borders
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Font size
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=9)

# =========================
# 4. Main title and layout
# =========================
fig.suptitle(
    "Comparison of Average Training Time Across Different Models",
    fontsize=18,
    fontweight="bold",
    y=0.98
)

plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.savefig(
    "training_time_comparison_four_datasets.pdf",
    dpi=300,
    bbox_inches="tight"
)
#plt.show()