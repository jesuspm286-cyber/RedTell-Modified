import argparse
import os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True)
args = parser.parse_args()

DATA_DIR = args.data

file_path = os.path.join(DATA_DIR, "hierarchical_analysis_table.csv")

df = pd.read_csv(file_path)

# Extract condition info from image name
names = df["image"].apply(lambda x: Path(str(x)).stem)

####### Pattern for parsing image names #########

### For colored images ###

# pattern = (
#     r"RBCs_(?P<plastic>PE|PP|PS)_"
#     # r"(?:_NPs)?_"
#     r"(?P<concentration>100|20|50|5|control)_"
#     r"(?P<picture>\d+)_"
#     r"(?P<time>\d+)hrs_"
#     r"(?P<quarter>\d+)$"    
# )


### For gray-scale images ###

pattern = (
    r"^(?P<plastic>PE|PP|PS)_RBCs"
    r"(?:_NPs)?_"
    r"(?P<concentration>control|1-\d+)_"
    r"(?P<picture>\d+)_"
    r"(?P<time>\d+)hrs_"
    r"(?P<quarter>\d+)$"
)

condition_info = names.str.extract(pattern)
df = pd.concat([df, condition_info], axis=1)

# Optional: check if any filenames failed to parse
bad_rows = df[df["plastic"].isna()]
if len(bad_rows) > 0:
    print("Some image names did not match the expected pattern:")
    print(bad_rows["image"].unique())

# Count cells by plastic, concentration, time, and prediction
summary_counts = (
    df.groupby(["plastic", "concentration", "time", "final_prediction"])
    #df.groupby(["plastic", "concentration", "time", "y_pred"])
    # df.groupby(["plastic", "concentration", "time", "predicted_label"])
      .size()
      .unstack(fill_value=0)
      .reset_index()
)

# Make sure all expected labels exist as columns
for label in ["Healthy", "Stressed", "Crenated", "Dead"]:
    if label not in summary_counts.columns:
        summary_counts[label] = 0

# Add total number of cells
summary_counts["Total"] = (
    summary_counts["Healthy"] +
    summary_counts["Stressed"] +
    summary_counts["Crenated"] +
    summary_counts["Dead"]
)

# Reorder columns
summary_counts = summary_counts[
    [
        "plastic",
        "concentration",
        "time",
        "Healthy",
        "Stressed",
        "Crenated",
        "Dead",
        "Total",
    ]
]

print(summary_counts)

# Save result
summary_counts.to_csv(os.path.join(DATA_DIR, "cell_counts_by_condition.csv"), index=False)


# Variables for graphing/analysis
healthy_counts = summary_counts["Healthy"].values
stressed_counts = summary_counts["Stressed"].values
crenated_counts = summary_counts["Crenated"].values
dead_counts = summary_counts["Dead"].values
total_counts = summary_counts["Total"].values

plastics = summary_counts["plastic"].values
concentrations = summary_counts["concentration"].values
times = summary_counts["time"].values

# Also useful: dictionary version
counts_by_condition = {}

for _, row in summary_counts.iterrows():
    key = (row["plastic"], row["concentration"], row["time"])
    counts_by_condition[key] = {
        "Healthy": row["Healthy"],
        "Stressed": row["Stressed"],
        "Crenated": row["Crenated"],
        "Dead": row["Dead"],
        "Total": row["Total"],
    }


# --------------------------------------------
# Percentages
# --------------------------------------------

summary_counts["Healthy_%"] = (
    summary_counts["Healthy"] / summary_counts["Total"] * 100
)

summary_counts["Stressed_%"] = (
    summary_counts["Stressed"] / summary_counts["Total"] * 100
)

summary_counts["Crenated_%"] = (
    summary_counts["Crenated"] / summary_counts["Total"] * 100
)

summary_counts["Dead_%"] = (
    summary_counts["Dead"] / summary_counts["Total"] * 100
)

summary_counts["actual_healthy_%"] = (
    (summary_counts["Total"] - summary_counts["Dead"] - summary_counts["Crenated"]) / summary_counts["Total"] * 100
)

summary_counts.to_csv(
    os.path.join(DATA_DIR, "cell_percentages_by_condition.csv"),
    index=False
)

print(summary_counts)

######### Plot Percentages #########

time = "15" # Change this to the desired time point (e.g., "3" or "15")

plot_df = summary_counts[
    summary_counts["time"] == time
].copy()

# plot_df["concentration_numeric"] = (
#     plot_df["concentration"]
#     .replace("control", "0")
#     .astype(float)
# )

concentration_map = {
    "control": 0,
    "1-200": 5,
    "1-50": 20,
    "1-20": 50,
    "1-10": 100,
    "5": 5,
    "20": 20,
    "50": 50,
    "100": 100,
}

plot_df["concentration_numeric"] = (
    plot_df["concentration"]
    .map(concentration_map)
)

print(plot_df[["plastic", "concentration", "concentration_numeric"]].drop_duplicates())

errors = {
    "PE": 0.5,
    "PP": 0.8,
    "PS": 0.6,
}

plt.figure(figsize=(10,7))

for plastic in ["PE", "PP", "PS"]:

    d = plot_df[
        plot_df["plastic"] == plastic
    ].sort_values("concentration_numeric")

    plt.errorbar(
        d["concentration_numeric"],
        d["Crenated_%"],
        yerr=0,
        marker="o",
        linewidth=2,
        capsize=5,
        label=plastic
    )
    # plt.errorbar(
    #     d["concentration_numeric"],
    #     d["actual_healthy_%"],
    #     yerr=4,
    #     marker="o",
    #     linewidth=2,
    #     capsize=5,
    #     label=plastic
    # )

plt.xticks([0, 5, 20, 50, 100], fontsize=14)
# plt.axhspan(
#     50.5, 66.1,
#     color='orange',
#     alpha=0.3,
#     label='Natural death'
# )

plt.xlabel('Nanoplastics Concentrations (μg/mL)', fontsize=20)
plt.ylabel('Percentage of Crenated cells (%)', fontsize=20)
plt.ylim(0, 10)
plt.title('Relative percentage of Crenated RBCs with nanoplastics \n at different concentrations after 15 hours of incubation',
            fontsize=17)
plt.legend(fontsize=15)
plt.tight_layout()
plt.show()
# plt.savefig(os.path.join(DATA_DIR, 'Crenated RBCs with NPs at different concentrations after 15 hours.png'),
#               dpi=300, bbox_inches='tight')