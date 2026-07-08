import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

file_path = "Colored_data/hierarchical_analysis_table.csv"
# file_path = "Data_MC/inference_predictions.csv"
df = pd.read_csv(file_path)

# Extract condition info from image name
names = df["image"].apply(lambda x: Path(str(x)).stem)



pattern = (
    r"RBCs_(?P<plastic>PE|PP|PS)_"
    # r"(?:_NPs)?_"
    r"(?P<concentration>100|20|50|5|control)_"
    r"(?P<picture>\d+)_"
    r"(?P<time>\d+)hrs_"
    r"(?P<quarter>\d+)$"    
)

# pattern = (
#     r"^(?P<plastic>PE|PP|PS)_RBCs"
#     r"(?:_NPs)?_"
#     r"(?P<concentration>control|1-\d+)_"
#     r"(?P<picture>\d+)_"
#     r"(?P<time>\d+)hrs_"
#     r"(?P<quarter>\d+)$"
# )

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
    # df.groupby(["plastic", "concentration", "time", "y_pred"])
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
# summary_counts.to_csv("Colored_data/classification_results_reviewed/cell_counts_by_condition.csv", index=False)
summary_counts.to_csv("Colored_data/cell_counts_by_condition.csv", index=False)

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


# def parse_condition(image_path):
#     name = os.path.basename(image_path)
#     base = os.path.splitext(name)[0]
#     parts = base.split("_")

#     # RBCs_PS_control_9_15hrs_3
#     plastic = parts[1]
#     concentration = parts[2]
#     time = parts[4].replace("hrs", "")

#     return plastic, concentration, time

# print(counts_by_condition.keys())


# # Example:
# print(counts_by_condition[("RBCs", "PE", "100", "15")])
# print(counts_by_condition[("PE", "1-20", "15")])
# print(counts_by_condition[("PE", "1-50", "15")])
# print(counts_by_condition[("PE", "1-200", "15")])
# print(counts_by_condition[("PE", "control", "15")])

# PE_1_10_15hrs_dead = counts_by_condition[("PE", "100", "15")]["Dead"] / counts_by_condition[("PE", "100", "15")]["Total"] * 100
# PE_1_20_15hrs_dead = counts_by_condition[("PE", "50", "15")]["Dead"] / counts_by_condition[("PE", "50", "15")]["Total"] * 100
# PE_1_50_15hrs_dead = counts_by_condition[("PE", "20", "15")]["Dead"] / counts_by_condition[("PE", "20", "15")]["Total"] * 100
# PE_1_200_15hrs_dead = counts_by_condition[("PE", "5", "15")]["Dead"] / counts_by_condition[("PE", "5", "15")]["Total"] * 100
# PE_control_15hrs_dead = counts_by_condition[("PE", "control", "15")]["Dead"] / counts_by_condition[("PE", "control", "15")]["Total"] * 100

# PP_1_10_15hrs_dead = counts_by_condition[("PP", "100", "15")]["Dead"] / counts_by_condition[("PP", "100", "15")]["Total"] * 100
# PP_1_20_15hrs_dead = counts_by_condition[("PP", "50", "15")]["Dead"] / counts_by_condition[("PP", "50", "15")]["Total"] * 100
# PP_1_50_15hrs_dead = counts_by_condition[("PP", "20", "15")]["Dead"] / counts_by_condition[("PP", "20", "15")]["Total"] * 100
# PP_1_200_15hrs_dead = counts_by_condition[("PP", "5", "15")]["Dead"] / counts_by_condition[("PP", "5", "15")]["Total"] * 100
# PP_control_15hrs_dead = counts_by_condition[("PP", "control", "15")]["Dead"] / counts_by_condition[("PP", "control", "15")]["Total"] * 100

# PS_1_10_15hrs_dead = counts_by_condition[("PS", "100", "15")]["Dead"] / counts_by_condition[("PS", "100", "15")]["Total"] * 100
# PS_1_20_15hrs_dead = counts_by_condition[("PS", "50", "15")]["Dead"] / counts_by_condition[("PS", "50", "15")]["Total"] * 100
# PS_1_50_15hrs_dead = counts_by_condition[("PS", "20", "15")]["Dead"] / counts_by_condition[("PS", "20", "15")]["Total"] * 100
# PS_1_200_15hrs_dead = counts_by_condition[("PS", "5", "15")]["Dead"] / counts_by_condition[("PS", "5", "15")]["Total"] * 100
# PS_control_15hrs_dead = counts_by_condition[("PS", "control", "15")]["Dead"] / counts_by_condition[("PS", "control", "15")]["Total"] * 100



# PE_1_10_3hrs_dead = counts_by_condition[("PE", "100", "3")]["Dead"] / counts_by_condition[("PE", "100", "3")]["Total"] * 100
# PE_1_20_3hrs_dead = counts_by_condition[("PE", "50", "3")]["Dead"] / counts_by_condition[("PE", "50", "3")]["Total"] * 100
# PE_1_50_3hrs_dead = counts_by_condition[("PE", "20", "3")]["Dead"] / counts_by_condition[("PE", "20", "3")]["Total"] * 100
# PE_1_200_3hrs_dead = counts_by_condition[("PE", "5", "3")]["Dead"] / counts_by_condition[("PE", "5", "3")]["Total"] * 100
# PE_control_3hrs_dead = counts_by_condition[("PE", "control", "3")]["Dead"] / counts_by_condition[("PE", "control", "3")]["Total"] * 100

# PP_1_10_3hrs_dead = counts_by_condition[("PP", "100", "3")]["Dead"] / counts_by_condition[("PP", "100", "3")]["Total"] * 100
# PP_1_20_3hrs_dead = counts_by_condition[("PP", "50", "3")]["Dead"] / counts_by_condition[("PP", "50", "3")]["Total"] * 100
# PP_1_50_3hrs_dead = counts_by_condition[("PP", "20", "3")]["Dead"] / counts_by_condition[("PP", "20", "3")]["Total"] * 100
# PP_1_200_3hrs_dead = counts_by_condition[("PP", "5", "3")]["Dead"] / counts_by_condition[("PP", "5", "3")]["Total"] * 100
# PP_control_3hrs_dead = counts_by_condition[("PP", "control", "3")]["Dead"] / counts_by_condition[("PP", "control", "3")]["Total"] * 100

# PS_1_10_3hrs_dead = counts_by_condition[("PS", "100", "3")]["Dead"] / counts_by_condition[("PS", "100", "3")]["Total"] * 100
# PS_1_20_3hrs_dead = counts_by_condition[("PS", "50", "3")]["Dead"] / counts_by_condition[("PS", "50", "3")]["Total"] * 100
# PS_1_50_3hrs_dead = counts_by_condition[("PS", "20", "3")]["Dead"] / counts_by_condition[("PS", "20", "3")]["Total"] * 100
# PS_1_200_3hrs_dead = counts_by_condition[("PS", "5", "3")]["Dead"] / counts_by_condition[("PS", "5", "3")]["Total"] * 100
# PS_control_3hrs_dead = counts_by_condition[("PS", "control", "3")]["Dead"] / counts_by_condition[("PS", "control", "3")]["Total"] * 100


# Data_3hrs_PS = [PS_control_3hrs_dead, PS_1_200_3hrs_dead, PS_1_50_3hrs_dead, PS_1_20_3hrs_dead, PS_1_10_3hrs_dead]
# Data_15hrs_PS = [PS_control_15hrs_dead, PS_1_200_15hrs_dead, PS_1_50_15hrs_dead, PS_1_20_15hrs_dead, PS_1_10_15hrs_dead]

# Data_3hrs_PP = [PP_control_3hrs_dead, PP_1_200_3hrs_dead, PP_1_50_3hrs_dead, PP_1_20_3hrs_dead, PP_1_10_3hrs_dead]
# Data_15hrs_PP = [PP_control_15hrs_dead, PP_1_200_15hrs_dead, PP_1_50_15hrs_dead, PP_1_20_15hrs_dead, PP_1_10_15hrs_dead]

# Data_3hrs_PE = [PE_control_3hrs_dead, PE_1_200_3hrs_dead, PE_1_50_3hrs_dead, PE_1_20_3hrs_dead, PE_1_10_3hrs_dead]
# Data_15hrs_PE = [PE_control_15hrs_dead, PE_1_200_15hrs_dead, PE_1_50_15hrs_dead, PE_1_20_15hrs_dead, PE_1_10_15hrs_dead]

# Concentrations = [0, 5, 20, 50, 100]

# print (Data_3hrs_PE)
# print (Data_15hrs_PE)

# print (Data_3hrs_PP)
# print (Data_15hrs_PP)

# print (Data_3hrs_PS)
# print (Data_15hrs_PS)

# # Error_3hrs_PE = [Err_control_PE[1], Err_100_PE[1], Err_50_PE[1], Err_20_PE[1], Err_5_PE[1]]
# # Error_15hrs_PE = [Err_control_PE[0], Err_100_PE[0], Err_50_PE[0], Err_20_PE[0], Err_5_PE[0]]


# # Error_3hrs_PP = [Err_control_PP[1], Err_100_PP[1], Err_50_PP[1], Err_20_PP[1], Err_5_PP[1]]
# # Error_15hrs_PP = [Err_control_PP[0], Err_100_PP[0], Err_50_PP[0], Err_20_PP[0], Err_5_PP[0]]


# # Error_3hrs_PS = [Err_control_PS[1], Err_100_PS[1], Err_50_PS[1], Err_20_PS[1], Err_5_PS[1]]
# # Error_15hrs_PS = [Err_control_PS[0], Err_100_PS[0], Err_50_PS[0], Err_20_PS[0], Err_5_PS[0]]


# plt.figure(figsize=(13, 10))

# # plt.errorbar(
# #     Concentrations,
# #     Data_15hrs_PE,
# #     fmt='o-', markersize=8, linewidth=2,
# #     color='black',
# #     capsize=5,
# #     label='Exposure to PE'
# # )

# # plt.errorbar(
# #     Concentrations,
# #     Data_15hrs_PP,
# #     yerr=Error_15hrs_PP,
# #     fmt='o-', markersize=8, linewidth=2,
# #     color='red',
# #     capsize=5,
# #     label='Exposure to PP'
# # )

# # plt.errorbar(
# #     Concentrations,
# #     Data_15hrs_PS,
# #     yerr=Error_15hrs_PS,
# #     fmt='o-', markersize=8, linewidth=2,
# #     color='dodgerblue',
# #     capsize=5,
# #     label='Exposure to PS'
# # )

# plt.xticks([0, 5, 20, 50, 100], fontsize=14)
# plt.plot(Concentrations, Data_3hrs_PE,'o-', markersize=8, linewidth=2, color='black', label='Exposure to PE')
# plt.plot(Concentrations, Data_3hrs_PP,'o-', markersize=8, linewidth=2, color='red', label='Exposure to PP')
# plt.plot(Concentrations, Data_3hrs_PS, 'o-', markersize=8, linewidth=2, color='dodgerblue', label='Exposure to PS')

# plt.axhspan(
#     0.4, 1.4,
#     color='orange',
#     alpha=0.3,
#     label='Natural death'
# )
# plt.xlabel('Nanoplastics Concentrations (μg/mL)', fontsize=20)
# plt.ylabel('Percentage of death (%)', fontsize=20)
# plt.title('Relative percentage of dead RBCs with nanoplastics \n at different concentrations after 3 hours of incubation',
#             fontsize=17)
# plt.ylim(0, 8)
# plt.legend(fontsize=15)
# plt.grid(False)

# plt.savefig('Colored_data/Dead RBCs with NPs at different concentrations after 3 hours.png',
#              dpi=300, bbox_inches='tight')
# #plt.show()


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
    "Colored_data/cell_percentages_by_condition.csv",
    index=False
)

print(summary_counts)

######### Plot Percentages #########

time = "3" # Change this to the desired time point (e.g., "3" or "15")

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
        d["Dead_%"],
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
plt.axhspan(
    0, 1.1,
    color='orange',
    alpha=0.3,
    label='Natural death'
)

plt.xlabel('Nanoplastics Concentrations (μg/mL)', fontsize=20)
plt.ylabel('Percentage of dead cells (%)', fontsize=20)
plt.ylim(0, 10)
plt.title('Relative percentage of dead RBCs with nanoplastics \n at different concentrations after 3 hours of incubation',
            fontsize=17)
plt.legend(fontsize=15)
plt.tight_layout()
#plt.show()
plt.savefig('Colored_data/Dead RBCs with NPs at different concentrations after 3 hours_RedTell.png',
              dpi=300, bbox_inches='tight')