# ==========================================================
# XAI METHODS COMPARISON
# Breast Cancer Classification - Swin Transformer
#
# Methods:
# 1. Grad-CAM
# 2. Grad-CAM++
# 3. SHAP
# 4. LIME
# ==========================================================


# ==========================================================
# Imports
# ==========================================================

import os
import sys

import pandas as pd
import matplotlib.pyplot as plt


# ==========================================================
# Project Root
# ==========================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)


# ==========================================================
# Configuration
# ==========================================================

XAI_ROOT = os.path.join(
    PROJECT_ROOT,
    "results",
    "xai"
)


COMPARISON_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "xai_comparison"
)


# ==========================================================
# Create Results Directory
# ==========================================================

os.makedirs(
    COMPARISON_DIR,
    exist_ok=True
)


# ==========================================================
# XAI Methods
# ==========================================================

METHODS = {

    "Grad-CAM": "gradcam",

    "Grad-CAM++": "gradcam_plus_plus",

    "SHAP": "shap",

    "LIME": "lime"

}


# ==========================================================
# Categories
# ==========================================================

CATEGORIES = {

    "correct_benign": "Correct Benign",

    "correct_malignant": "Correct Malignant",

    "false_positive": "False Positive",

    "false_negative": "False Negative"

}


# ==========================================================
# Print Header
# ==========================================================

print()

print("=" * 70)

print("XAI METHODS COMPARISON")

print("=" * 70)

print()

print(
    "Project Root:",
    PROJECT_ROOT
)

print(
    "XAI Results Root:",
    XAI_ROOT
)

print(
    "Comparison Results:",
    COMPARISON_DIR
)

print()

print("=" * 70)


# ==========================================================
# Function:
# Count PNG Images
# ==========================================================

def count_png_files(
    folder_path
):

    if not os.path.exists(
        folder_path
    ):

        return 0


    files = [

        file

        for file in os.listdir(
            folder_path
        )

        if file.lower().endswith(
            ".png"
        )

    ]


    return len(
        files
    )


# ==========================================================
# Store Comparison Results
# ==========================================================

comparison_results = []


# ==========================================================
# Analyze Every XAI Method
# ==========================================================

print()

print(
    "ANALYZING XAI OUTPUT FOLDERS..."
)

print()


for method_name, folder_name in METHODS.items():

    print(
        "-" * 70
    )

    print(
        "Method:",
        method_name
    )

    print(
        "-" * 70
    )


    method_path = os.path.join(

        XAI_ROOT,

        folder_name

    )


    # ------------------------------------------------------
    # Check Method Folder
    # ------------------------------------------------------

    if not os.path.exists(
        method_path
    ):

        print()

        print(
            "WARNING: Folder not found!"
        )

        print(
            method_path
        )


    # ------------------------------------------------------
    # Analyze Categories
    # ------------------------------------------------------

    for category_name, category_display in CATEGORIES.items():

        category_path = os.path.join(

            method_path,

            category_name

        )


        image_count = count_png_files(

            category_path

        )


        # --------------------------------------------------
        # Status
        # --------------------------------------------------

        if image_count == 10:

            status = "Complete"

        elif image_count > 0:

            status = "Partial"

        else:

            status = "Missing"


        # --------------------------------------------------
        # Store Result
        # --------------------------------------------------

        comparison_results.append(

            {

                "Method": method_name,

                "Category": category_display,

                "Folder": category_path,

                "Images_Found": image_count,

                "Expected_Images": 10,

                "Status": status

            }

        )


        # --------------------------------------------------
        # Print
        # --------------------------------------------------

        print(

            f"{category_display:20s} : "

            f"{image_count}/10 "

            f"[{status}]"

        )


    print()


# ==========================================================
# Create DataFrame
# ==========================================================

df = pd.DataFrame(

    comparison_results

)


# ==========================================================
# Save Detailed CSV
# ==========================================================

detailed_csv_path = os.path.join(

    COMPARISON_DIR,

    "xai_comparison_detailed.csv"

)


df.to_csv(

    detailed_csv_path,

    index=False

)


print()

print("=" * 70)

print(
    "DETAILED COMPARISON SAVED"
)

print("=" * 70)

print()

print(
    detailed_csv_path
)


# ==========================================================
# Create Pivot Table
# ==========================================================

pivot_table = df.pivot(

    index="Method",

    columns="Category",

    values="Images_Found"

)


# ==========================================================
# Add Total Column
# ==========================================================

pivot_table["Total Images"] = pivot_table.sum(

    axis=1

)


# ==========================================================
# Reset Index
# ==========================================================

pivot_table = pivot_table.reset_index()


# ==========================================================
# Save Summary CSV
# ==========================================================

summary_csv_path = os.path.join(

    COMPARISON_DIR,

    "xai_comparison_summary.csv"

)


pivot_table.to_csv(

    summary_csv_path,

    index=False

)


# ==========================================================
# Print Summary Table
# ==========================================================

print()

print("=" * 70)

print(
    "XAI COMPARISON SUMMARY"
)

print("=" * 70)

print()

print(

    pivot_table.to_string(
        index=False
    )

)


# ==========================================================
# Calculate Method Totals
# ==========================================================

method_totals = df.groupby(

    "Method"

)[

    "Images_Found"

].sum()


# ==========================================================
# Print Total Images
# ==========================================================

print()

print("=" * 70)

print(
    "TOTAL EXPLANATIONS GENERATED"
)

print("=" * 70)

print()


for method, total in method_totals.items():

    print(

        f"{method:15s} : "

        f"{total} Images"

    )


grand_total = int(

    method_totals.sum()

)


print()

print(
    "-" * 70
)

print(

    f"{'TOTAL':15s} : "

    f"{grand_total} Images"

)

print(
    "-" * 70
)


# ==========================================================
# Generate Bar Chart
# ==========================================================

print()

print(
    "Generating Comparison Chart..."
)


# ----------------------------------------------------------
# Set Index
# ----------------------------------------------------------

chart_data = pivot_table.set_index(

    "Method"

)


# ----------------------------------------------------------
# Remove Total Column
# ----------------------------------------------------------

category_columns = [

    "Correct Benign",

    "Correct Malignant",

    "False Positive",

    "False Negative"

]


# Keep only available columns
available_columns = [

    column

    for column in category_columns

    if column in chart_data.columns

]


# ----------------------------------------------------------
# Plot
# ----------------------------------------------------------

ax = chart_data[
    available_columns
].plot(

    kind="bar",

    figsize=(12, 7)

)


# ----------------------------------------------------------
# Labels
# ----------------------------------------------------------

plt.title(

    "XAI Methods Comparison by Prediction Category",

    fontsize=15,

    fontweight="bold"

)


plt.xlabel(

    "XAI Method",

    fontsize=12

)


plt.ylabel(

    "Number of Explanation Images",

    fontsize=12

)


plt.xticks(

    rotation=0

)


plt.legend(

    title="Prediction Category"

)


plt.tight_layout()


# ==========================================================
# Save Chart
# ==========================================================

chart_path = os.path.join(

    COMPARISON_DIR,

    "xai_methods_comparison.png"

)


plt.savefig(

    chart_path,

    dpi=300,

    bbox_inches="tight"

)


plt.close()


print()

print(
    "Chart Saved:"
)

print(
    chart_path
)


# ==========================================================
# Completeness Analysis
# ==========================================================

print()

print("=" * 70)

print(
    "COMPLETENESS CHECK"
)

print("=" * 70)

print()


complete_count = len(

    df[
        df["Status"] == "Complete"
    ]

)


partial_count = len(

    df[
        df["Status"] == "Partial"
    ]

)


missing_count = len(

    df[
        df["Status"] == "Missing"
    ]

)


total_checks = len(
    df
)


print(

    "Complete Categories:",

    complete_count,

    "/",

    total_checks

)


print(

    "Partial Categories:",

    partial_count

)


print(

    "Missing Categories:",

    missing_count

)


# ==========================================================
# Final Status
# ==========================================================

print()

print("=" * 70)


if (

    complete_count == total_checks

):

    print(

        "ALL XAI RESULTS VERIFIED SUCCESSFULLY!"

    )

else:

    print(

        "WARNING: SOME XAI RESULTS ARE MISSING OR PARTIAL"

    )


print("=" * 70)


# ==========================================================
# Final Output Information
# ==========================================================

print()

print(
    "FILES GENERATED:"
)

print()

print(

    "1.",

    detailed_csv_path

)

print(

    "2.",

    summary_csv_path

)

print(

    "3.",

    chart_path

)


print()

print("=" * 70)

print(
    "XAI COMPARISON COMPLETED SUCCESSFULLY!"
)

print("=" * 70)