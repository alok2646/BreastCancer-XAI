import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# FINAL XAI RESULTS ANALYSIS
# ============================================================

print("\n" + "=" * 75)
print("FINAL XAI RESULTS ANALYSIS")
print("=" * 75)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

QUANTITATIVE_DIR = PROJECT_ROOT / "results" / "xai_quantitative"

OUTPUT_DIR = PROJECT_ROOT / "results" / "final_xai_analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


print(f"\nProject Root: {PROJECT_ROOT}")
print(f"Quantitative Results: {QUANTITATIVE_DIR}")
print(f"Output Directory: {OUTPUT_DIR}")


# ============================================================
# INPUT FILE PATHS
# ============================================================

SUMMARY_FILE = (
    QUANTITATIVE_DIR /
    "xai_quantitative_summary.csv"
)

CATEGORY_FILE = (
    QUANTITATIVE_DIR /
    "xai_quantitative_category_summary.csv"
)

RANKING_FILE = (
    QUANTITATIVE_DIR /
    "xai_overall_ranking.csv"
)


# ============================================================
# CHECK FILES
# ============================================================

required_files = [
    SUMMARY_FILE,
    CATEGORY_FILE,
    RANKING_FILE
]

for file_path in required_files:

    if not file_path.exists():

        print("\nERROR: Required file not found!")

        print(file_path)

        exit()


print("\nAll required quantitative files found successfully!")


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 75)
print("LOADING QUANTITATIVE RESULTS")
print("=" * 75)


summary_df = pd.read_csv(
    SUMMARY_FILE
)

category_df = pd.read_csv(
    CATEGORY_FILE
)

ranking_df = pd.read_csv(
    RANKING_FILE
)


print("\nMethod Summary:")

print(
    summary_df.to_string(
        index=False
    )
)


print("\nOverall Ranking:")

print(
    ranking_df.to_string(
        index=False
    )
)


# ============================================================
# ROUND VALUES
# ============================================================

numeric_columns = [

    "Concentration",
    "Sparsity",
    "Activation",
    "Max_Importance",
    "Top10_Importance",
    "Explanation_Area"

]


for column in numeric_columns:

    if column in summary_df.columns:

        summary_df[column] = (
            summary_df[column]
            .round(6)
        )


ranking_df[
    "Overall_XAI_Score"
] = (

    ranking_df[
        "Overall_XAI_Score"
    ]
    .round(6)

)


# ============================================================
# ADD RANK
# ============================================================

ranking_df.insert(

    0,

    "Rank",

    range(
        1,
        len(ranking_df) + 1
    )

)


# ============================================================
# SAVE RESEARCH PAPER TABLE 1
# ============================================================

table1_path = (

    OUTPUT_DIR /

    "table1_xai_method_comparison.csv"

)


summary_df.to_csv(

    table1_path,

    index=False

)


print("\n" + "=" * 75)

print(
    "TABLE 1: XAI METHOD COMPARISON SAVED"
)

print("=" * 75)

print(
    table1_path
)


# ============================================================
# SAVE RESEARCH PAPER TABLE 2
# ============================================================

table2_path = (

    OUTPUT_DIR /

    "table2_xai_overall_ranking.csv"

)


ranking_df.to_csv(

    table2_path,

    index=False

)


print("\n" + "=" * 75)

print(
    "TABLE 2: OVERALL XAI RANKING SAVED"
)

print("=" * 75)

print(
    table2_path
)


# ============================================================
# CATEGORY-WISE ANALYSIS
# ============================================================

print("\n" + "=" * 75)

print(
    "CATEGORY-WISE XAI ANALYSIS"
)

print("=" * 75)


category_metrics = [

    "Concentration",
    "Sparsity",
    "Activation",
    "Explanation_Area"

]


category_analysis = (

    category_df[

        [
            "Method",
            "Category"
        ]

        +

        category_metrics

    ]

)


print("\n")

print(

    category_analysis.to_string(

        index=False

    )

)


# ============================================================
# SAVE CATEGORY TABLE
# ============================================================

table3_path = (

    OUTPUT_DIR /

    "table3_category_wise_analysis.csv"

)


category_analysis.to_csv(

    table3_path,

    index=False

)


print("\nCategory-wise table saved:")

print(
    table3_path
)


# ============================================================
# BEST METHOD ANALYSIS
# ============================================================

best_method = (

    ranking_df.iloc[0]

)


print("\n" + "=" * 75)

print(
    "BEST PERFORMING XAI METHOD"
)

print("=" * 75)


print(

    f"\nRank: {best_method['Rank']}"

)

print(

    f"Method: {best_method['Method']}"

)

print(

    f"Overall XAI Score: "
    f"{best_method['Overall_XAI_Score']:.6f}"

)


# ============================================================
# CALCULATE BEST METHOD FOR EACH METRIC
# ============================================================

metric_results = []


for metric in numeric_columns:

    if metric not in summary_df.columns:

        continue


    best_row = (

        summary_df.loc[

            summary_df[metric].idxmax()

        ]

    )


    metric_results.append(

        {

            "Metric":

                metric,

            "Best_Method":

                best_row["Method"],

            "Best_Score":

                best_row[metric]

        }

    )


metric_best_df = pd.DataFrame(

    metric_results

)


metric_best_path = (

    OUTPUT_DIR /

    "table4_best_method_per_metric.csv"

)


metric_best_df.to_csv(

    metric_best_path,

    index=False

)


print("\n" + "=" * 75)

print(
    "BEST METHOD FOR EACH QUANTITATIVE METRIC"
)

print("=" * 75)


print("\n")

print(

    metric_best_df.to_string(

        index=False

    )

)


# ============================================================
# GENERATE RANKING GRAPH
# ============================================================

print("\n" + "=" * 75)

print(
    "GENERATING RESEARCH PAPER FIGURES"
)

print("=" * 75)


plt.figure(

    figsize=(10, 6)

)


plt.bar(

    ranking_df["Method"],

    ranking_df["Overall_XAI_Score"]

)


plt.title(

    "Overall Quantitative Comparison of XAI Methods"

)


plt.xlabel(

    "XAI Method"

)


plt.ylabel(

    "Overall XAI Score"

)


plt.tight_layout()


ranking_graph = (

    OUTPUT_DIR /

    "figure1_xai_overall_ranking.png"

)


plt.savefig(

    ranking_graph,

    dpi=300,

    bbox_inches="tight"

)


plt.close()


print(

    f"\nFigure 1 Saved: {ranking_graph}"

)


# ============================================================
# METRIC COMPARISON FIGURE
# ============================================================

plot_metrics = [

    "Concentration",
    "Sparsity",
    "Activation",
    "Explanation_Area"

]


available_metrics = [

    metric

    for metric in plot_metrics

    if metric in summary_df.columns

]


summary_plot = (

    summary_df.set_index(

        "Method"

    )[

        available_metrics

    ]

)


summary_plot.plot(

    kind="bar",

    figsize=(12, 7)

)


plt.title(

    "Quantitative Metric Comparison Across XAI Methods"

)


plt.xlabel(

    "XAI Method"

)


plt.ylabel(

    "Metric Score"

)


plt.xticks(

    rotation=0

)


plt.legend(

    title="Metrics"

)


plt.tight_layout()


metric_graph = (

    OUTPUT_DIR /

    "figure2_quantitative_metric_comparison.png"

)


plt.savefig(

    metric_graph,

    dpi=300,

    bbox_inches="tight"

)


plt.close()


print(

    f"Figure 2 Saved: {metric_graph}"

)


# ============================================================
# CATEGORY-WISE FIGURE
# ============================================================

pivot_df = (

    category_df.pivot(

        index="Category",

        columns="Method",

        values="Concentration"

    )

)


pivot_df.plot(

    kind="bar",

    figsize=(12, 7)

)


plt.title(

    "Category-Wise Concentration Comparison"

)


plt.xlabel(

    "Prediction Category"

)


plt.ylabel(

    "Concentration Score"

)


plt.xticks(

    rotation=0

)


plt.legend(

    title="XAI Method"

)


plt.tight_layout()


category_graph = (

    OUTPUT_DIR /

    "figure3_category_wise_concentration.png"

)


plt.savefig(

    category_graph,

    dpi=300,

    bbox_inches="tight"

)


plt.close()


print(

    f"Figure 3 Saved: {category_graph}"

)


# ============================================================
# GENERATE TEXTUAL RESEARCH FINDINGS
# ============================================================

findings_path = (

    OUTPUT_DIR /

    "research_findings.txt"

)


with open(

    findings_path,

    "w",

    encoding="utf-8"

) as file:


    file.write(

        "=" * 75 + "\n"

    )

    file.write(

        "FINAL XAI RESULTS ANALYSIS\n"

    )

    file.write(

        "=" * 75 + "\n\n"

    )


    file.write(

        "1. OVERALL XAI METHOD RANKING\n\n"

    )


    for _, row in ranking_df.iterrows():

        file.write(

            f"Rank {row['Rank']}: "

            f"{row['Method']} "

            f"(Overall Score: "

            f"{row['Overall_XAI_Score']:.6f})\n"

        )


    file.write(

        "\n"

    )


    file.write(

        "2. BEST METHOD PER QUANTITATIVE METRIC\n\n"

    )


    for _, row in metric_best_df.iterrows():

        file.write(

            f"{row['Metric']}: "

            f"{row['Best_Method']} "

            f"(Score: "

            f"{row['Best_Score']:.6f})\n"

        )


    file.write(

        "\n"

    )


    file.write(

        "3. OVERALL FINDING\n\n"

    )


    file.write(

        f"The quantitative evaluation indicates that "

        f"{best_method['Method']} achieved the highest "

        f"overall XAI score of "

        f"{best_method['Overall_XAI_Score']:.6f} "

        f"among the evaluated explanation methods.\n"

    )


    file.write(

        "\n"

    )


    file.write(

        "The evaluation compared Grad-CAM, Grad-CAM++, "

        "SHAP, and LIME across concentration, sparsity, "

        "activation, maximum importance, top importance, "

        "and explanation area metrics.\n"

    )


print(

    f"\nResearch Findings Saved: {findings_path}"

)


# ============================================================
# FINAL OUTPUT SUMMARY
# ============================================================

print("\n" + "=" * 75)

print(
    "FINAL XAI ANALYSIS COMPLETED SUCCESSFULLY!"
)

print("=" * 75)


print(

    "\nFILES GENERATED:"

)


print(

    f"\n1. {table1_path}"

)

print(

    f"2. {table2_path}"

)

print(

    f"3. {table3_path}"

)

print(

    f"4. {metric_best_path}"

)

print(

    f"5. {ranking_graph}"

)

print(

    f"6. {metric_graph}"

)

print(

    f"7. {category_graph}"

)

print(

    f"8. {findings_path}"

)


print(

    "\n" + "=" * 75

)

print(

    "RESEARCH PAPER READY XAI RESULTS GENERATED!"

)

print("=" * 75 + "\n")