import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.stats import entropy


# ============================================================
# XAI QUANTITATIVE EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("XAI QUANTITATIVE EVALUATION")
print("=" * 70)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

XAI_ROOT = PROJECT_ROOT / "results" / "xai"

OUTPUT_DIR = PROJECT_ROOT / "results" / "xai_quantitative"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


print(f"\nProject Root: {PROJECT_ROOT}")
print(f"XAI Root: {XAI_ROOT}")
print(f"Output Directory: {OUTPUT_DIR}")


# ============================================================
# XAI METHODS
# ============================================================

METHODS = {
    "Grad-CAM": XAI_ROOT / "gradcam",
    "Grad-CAM++": XAI_ROOT / "gradcam_plus_plus",
    "SHAP": XAI_ROOT / "shap",
    "LIME": XAI_ROOT / "lime"
}


# ============================================================
# CATEGORIES
# ============================================================

CATEGORIES = [
    "correct_benign",
    "correct_malignant",
    "false_positive",
    "false_negative"
]


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(image_path):

    image = cv2.imread(str(image_path))

    if image is None:
        return None

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    return image


# ============================================================
# CREATE APPROXIMATE HEATMAP
# ============================================================

def extract_heatmap(image):

    """
    Convert explanation visualization into
    an approximate importance map.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    gray = gray.astype(np.float32)

    # Normalize
    gray = gray - np.min(gray)

    max_value = np.max(gray)

    if max_value > 0:
        gray = gray / max_value

    return gray


# ============================================================
# CONCENTRATION SCORE
# ============================================================

def concentration_score(heatmap):

    """
    Measures how concentrated the explanation is.

    Higher score = more focused explanation.
    """

    flat = heatmap.flatten()

    flat = flat[flat > 0]

    if len(flat) == 0:
        return 0

    total = np.sum(flat)

    if total == 0:
        return 0

    probabilities = flat / total

    entropy_value = entropy(probabilities)

    max_entropy = np.log(len(probabilities))

    if max_entropy == 0:
        return 0

    concentration = 1 - (
        entropy_value / max_entropy
    )

    return concentration


# ============================================================
# SPARSITY SCORE
# ============================================================

def sparsity_score(heatmap):

    """
    Measures how sparse the explanation is.

    Higher score = fewer important pixels.
    """

    threshold = np.mean(heatmap)

    important_pixels = np.sum(
        heatmap > threshold
    )

    total_pixels = heatmap.size

    sparsity = 1 - (
        important_pixels / total_pixels
    )

    return sparsity


# ============================================================
# ACTIVATION SCORE
# ============================================================

def activation_score(heatmap):

    """
    Measures average importance intensity.
    """

    return float(np.mean(heatmap))


# ============================================================
# MAXIMUM IMPORTANCE
# ============================================================

def max_importance_score(heatmap):

    return float(np.max(heatmap))


# ============================================================
# TOP 10% IMPORTANCE SCORE
# ============================================================

def top_importance_score(heatmap):

    flat = heatmap.flatten()

    if len(flat) == 0:
        return 0

    top_percentage = int(
        len(flat) * 0.10
    )

    if top_percentage == 0:
        top_percentage = 1

    top_values = np.sort(flat)[-top_percentage:]

    return float(np.mean(top_values))


# ============================================================
# EXPLANATION AREA SCORE
# ============================================================

def explanation_area_score(heatmap):

    """
    Percentage of pixels considered important.
    """

    threshold = np.mean(heatmap)

    important_pixels = np.sum(
        heatmap >= threshold
    )

    total_pixels = heatmap.size

    score = important_pixels / total_pixels

    return score


# ============================================================
# PROCESS SINGLE IMAGE
# ============================================================

def evaluate_image(image_path):

    image = load_image(image_path)

    if image is None:

        return {
            "Concentration": 0,
            "Sparsity": 0,
            "Activation": 0,
            "Max_Importance": 0,
            "Top10_Importance": 0,
            "Explanation_Area": 0
        }

    heatmap = extract_heatmap(image)

    results = {

        "Concentration":
            concentration_score(heatmap),

        "Sparsity":
            sparsity_score(heatmap),

        "Activation":
            activation_score(heatmap),

        "Max_Importance":
            max_importance_score(heatmap),

        "Top10_Importance":
            top_importance_score(heatmap),

        "Explanation_Area":
            explanation_area_score(heatmap)
    }

    return results


# ============================================================
# FIND IMAGE FILES
# ============================================================

def get_image_files(folder):

    extensions = [
        "*.png",
        "*.jpg",
        "*.jpeg"
    ]

    files = []

    for extension in extensions:

        files.extend(
            list(folder.glob(extension))
        )

    return sorted(files)


# ============================================================
# STORE RESULTS
# ============================================================

all_results = []


# ============================================================
# START EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("STARTING QUANTITATIVE EVALUATION")
print("=" * 70)


for method_name, method_path in METHODS.items():

    print("\n" + "-" * 70)

    print(
        f"\nEvaluating Method: {method_name}"
    )

    print("-" * 70)

    if not method_path.exists():

        print(
            f"WARNING: Folder not found: {method_path}"
        )

        continue


    for category in CATEGORIES:

        category_path = (
            method_path / category
        )

        print(
            f"\nCategory: {category}"
        )

        if not category_path.exists():

            print(
                f"WARNING: Category not found: {category_path}"
            )

            continue


        image_files = get_image_files(
            category_path
        )

        print(
            f"Images Found: {len(image_files)}"
        )


        for index, image_path in enumerate(
            image_files,
            start=1
        ):

            print(
                f"Processing Image {index}: "
                f"{image_path.name}"
            )

            metrics = evaluate_image(
                image_path
            )

            row = {

                "Method":
                    method_name,

                "Category":
                    category,

                "Image":
                    image_path.name,

                "Concentration":
                    metrics["Concentration"],

                "Sparsity":
                    metrics["Sparsity"],

                "Activation":
                    metrics["Activation"],

                "Max_Importance":
                    metrics["Max_Importance"],

                "Top10_Importance":
                    metrics["Top10_Importance"],

                "Explanation_Area":
                    metrics["Explanation_Area"]
            }

            all_results.append(row)


# ============================================================
# CREATE DATAFRAME
# ============================================================

print("\n" + "=" * 70)
print("CREATING RESULTS DATAFRAME")
print("=" * 70)


df = pd.DataFrame(
    all_results
)


if df.empty:

    print(
        "\nERROR: No XAI images found!"
    )

    print(
        "\nPlease check your XAI folder paths."
    )

    exit()


# ============================================================
# SAVE DETAILED RESULTS
# ============================================================

detailed_csv = (
    OUTPUT_DIR /
    "xai_quantitative_detailed.csv"
)


df.to_csv(
    detailed_csv,
    index=False
)


print(
    f"\nDetailed Results Saved:"
)

print(
    detailed_csv
)


# ============================================================
# METHOD-WISE SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("METHOD-WISE QUANTITATIVE SUMMARY")
print("=" * 70)


method_summary = (

    df.groupby("Method")[

        [
            "Concentration",
            "Sparsity",
            "Activation",
            "Max_Importance",
            "Top10_Importance",
            "Explanation_Area"
        ]

    ]

    .mean()

    .reset_index()

)


print("\n")

print(
    method_summary.to_string(
        index=False
    )
)


# ============================================================
# SAVE METHOD SUMMARY
# ============================================================

summary_csv = (
    OUTPUT_DIR /
    "xai_quantitative_summary.csv"
)


method_summary.to_csv(
    summary_csv,
    index=False
)


print(
    f"\nSummary Saved:"
)

print(
    summary_csv
)


# ============================================================
# CATEGORY-WISE SUMMARY
# ============================================================

category_summary = (

    df.groupby(

        [
            "Method",
            "Category"
        ]

    )[

        [
            "Concentration",
            "Sparsity",
            "Activation",
            "Max_Importance",
            "Top10_Importance",
            "Explanation_Area"
        ]

    ]

    .mean()

    .reset_index()

)


category_csv = (

    OUTPUT_DIR /
    "xai_quantitative_category_summary.csv"
)


category_summary.to_csv(

    category_csv,

    index=False
)


print(
    f"\nCategory Summary Saved:"
)

print(
    category_csv
)


# ============================================================
# GENERATE COMPARISON CHARTS
# ============================================================

print("\n" + "=" * 70)
print("GENERATING QUANTITATIVE COMPARISON CHARTS")
print("=" * 70)


metrics_to_plot = [

    "Concentration",
    "Sparsity",
    "Activation",
    "Max_Importance",
    "Top10_Importance",
    "Explanation_Area"

]


for metric in metrics_to_plot:

    print(
        f"\nGenerating Chart: {metric}"
    )


    plt.figure(

        figsize=(10, 6)

    )


    plt.bar(

        method_summary["Method"],

        method_summary[metric]

    )


    plt.title(

        f"XAI Methods Comparison - {metric}"

    )


    plt.xlabel(

        "XAI Method"

    )


    plt.ylabel(

        metric

    )


    plt.xticks(

        rotation=0

    )


    plt.tight_layout()


    chart_path = (

        OUTPUT_DIR /

        f"{metric.lower()}_comparison.png"

    )


    plt.savefig(

        chart_path,

        dpi=300,

        bbox_inches="tight"

    )


    plt.close()


    print(

        f"Saved: {chart_path}"

    )


# ============================================================
# OVERALL SCORE CALCULATION
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING OVERALL QUANTITATIVE SCORE")
print("=" * 70)


overall_scores = []


for _, row in method_summary.iterrows():

    concentration = row[
        "Concentration"
    ]

    sparsity = row[
        "Sparsity"
    ]

    activation = row[
        "Activation"
    ]

    top_importance = row[
        "Top10_Importance"
    ]


    overall_score = (

        concentration * 0.30

        +

        sparsity * 0.25

        +

        activation * 0.20

        +

        top_importance * 0.25

    )


    overall_scores.append(

        {

            "Method":
                row["Method"],

            "Overall_XAI_Score":
                overall_score

        }

    )


overall_df = pd.DataFrame(

    overall_scores

)


overall_df = (

    overall_df

    .sort_values(

        by="Overall_XAI_Score",

        ascending=False

    )

    .reset_index(

        drop=True

    )

)


print("\n")

print(

    overall_df.to_string(

        index=False

    )

)


# ============================================================
# SAVE OVERALL SCORE
# ============================================================

overall_csv = (

    OUTPUT_DIR /

    "xai_overall_ranking.csv"

)


overall_df.to_csv(

    overall_csv,

    index=False

)


print(

    f"\nOverall Ranking Saved:"

)

print(

    overall_csv

)


# ============================================================
# OVERALL RANKING CHART
# ============================================================

plt.figure(

    figsize=(10, 6)

)


plt.bar(

    overall_df["Method"],

    overall_df["Overall_XAI_Score"]

)


plt.title(

    "Overall XAI Method Ranking"

)


plt.xlabel(

    "XAI Method"

)


plt.ylabel(

    "Overall XAI Score"

)


plt.tight_layout()


ranking_chart = (

    OUTPUT_DIR /

    "overall_xai_ranking.png"

)


plt.savefig(

    ranking_chart,

    dpi=300,

    bbox_inches="tight"

)


plt.close()


print(

    f"\nRanking Chart Saved:"

)

print(

    ranking_chart

)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)

print(
    "XAI QUANTITATIVE EVALUATION COMPLETED SUCCESSFULLY!"
)

print("=" * 70)


print(

    f"\nTotal Images Evaluated: {len(df)}"

)


print(

    "\nMethods Evaluated:"

)


for method in df["Method"].unique():

    count = len(

        df[

            df["Method"] == method

        ]

    )

    print(

        f"{method}: {count} Images"

    )


print(

    "\n" + "=" * 70

)


print(

    "FILES GENERATED:"

)


print("=" * 70)


print(

    f"\n1. {detailed_csv}"

)

print(

    f"2. {summary_csv}"

)

print(

    f"3. {category_csv}"

)

print(

    f"4. {overall_csv}"

)

print(

    f"5. {ranking_chart}"

)


print(

    "\nAdditional metric comparison charts saved in:"

)

print(

    OUTPUT_DIR

)


print(

    "\n" + "=" * 70

)

print(

    "PROJECT XAI QUANTITATIVE ANALYSIS FINISHED!"

)

print("=" * 70 + "\n")