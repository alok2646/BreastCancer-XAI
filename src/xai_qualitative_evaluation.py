# ==========================================================
# XAI QUALITATIVE EVALUATION
# Breast Cancer Classification using Swin Transformer
#
# Comparison:
# Grad-CAM vs Grad-CAM++ vs SHAP vs LIME
# ==========================================================

import os
import glob
from PIL import Image

import matplotlib.pyplot as plt


# ==========================================================
# CONFIGURATION
# ==========================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

XAI_ROOT = os.path.join(
    PROJECT_ROOT,
    "results",
    "xai"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "xai_qualitative"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================================
# XAI METHODS
# ==========================================================

METHODS = {

    "Grad-CAM": os.path.join(
        XAI_ROOT,
        "gradcam"
    ),

    "Grad-CAM++": os.path.join(
        XAI_ROOT,
        "gradcam_plus_plus"
    ),

    "SHAP": os.path.join(
        XAI_ROOT,
        "shap"
    ),

    "LIME": os.path.join(
        XAI_ROOT,
        "lime"
    )

}


# ==========================================================
# CATEGORIES
# ==========================================================

CATEGORIES = [

    "correct_benign",

    "correct_malignant",

    "false_positive",

    "false_negative"

]


# ==========================================================
# FIND IMAGE
# ==========================================================

def find_image(method_path, category, index):

    category_path = os.path.join(
        method_path,
        category
    )

    patterns = [

        f"*_{index}.png",

        f"*{index}.png"

    ]

    for pattern in patterns:

        files = glob.glob(
            os.path.join(
                category_path,
                pattern
            )
        )

        if len(files) > 0:

            files.sort()

            return files[0]

    return None


# ==========================================================
# LOAD IMAGE
# ==========================================================

def load_image(image_path):

    try:

        image = Image.open(
            image_path
        ).convert("RGB")

        return image

    except Exception as error:

        print(
            "Error loading image:",
            image_path
        )

        print(error)

        return None


# ==========================================================
# CREATE COMPARISON FIGURE
# ==========================================================

def create_comparison(
    category,
    index
):

    print()

    print("=" * 70)

    print(
        f"Creating Comparison | "
        f"Category: {category} | "
        f"Sample: {index}"
    )

    print("=" * 70)


    # ------------------------------------------------------
    # CREATE FIGURE
    # ------------------------------------------------------

    figure, axes = plt.subplots(

        1,

        4,

        figsize=(20, 5)

    )


    figure.suptitle(

        f"XAI Method Comparison\n"
        f"Category: {category.replace('_', ' ').title()} | "
        f"Sample {index}",

        fontsize=18,

        fontweight="bold"

    )


    # ------------------------------------------------------
    # PROCESS METHODS
    # ------------------------------------------------------

    for axis, (
        method_name,
        method_path

    ) in zip(

        axes,

        METHODS.items()

    ):


        image_path = find_image(

            method_path,

            category,

            index

        )


        # --------------------------------------------------
        # IMAGE FOUND
        # --------------------------------------------------

        if image_path is not None:


            image = load_image(

                image_path

            )


            if image is not None:


                axis.imshow(

                    image

                )


                axis.set_title(

                    method_name,

                    fontsize=15,

                    fontweight="bold"

                )


            else:


                axis.text(

                    0.5,

                    0.5,

                    "Error Loading Image",

                    horizontalalignment="center",

                    verticalalignment="center"

                )


        # --------------------------------------------------
        # IMAGE NOT FOUND
        # --------------------------------------------------

        else:


            axis.text(

                0.5,

                0.5,

                "Image Not Found",

                horizontalalignment="center",

                verticalalignment="center",

                fontsize=13

            )


            axis.set_title(

                method_name,

                fontsize=15,

                fontweight="bold"

            )


            print(

                f"WARNING: Image not found | "
                f"{method_name} | "
                f"{category} | "
                f"Sample {index}"

            )


        # --------------------------------------------------
        # REMOVE AXIS
        # --------------------------------------------------

        axis.axis(

            "off"

        )


    # ------------------------------------------------------
    # ADJUST LAYOUT
    # ------------------------------------------------------

    plt.tight_layout(

        rect=[0, 0, 1, 0.90]

    )


    # ------------------------------------------------------
    # SAVE FIGURE
    # ------------------------------------------------------

    category_output = os.path.join(

        OUTPUT_DIR,

        category

    )


    os.makedirs(

        category_output,

        exist_ok=True

    )


    output_path = os.path.join(

        category_output,

        f"xai_comparison_{index}.png"

    )


    plt.savefig(

        output_path,

        dpi=300,

        bbox_inches="tight"

    )


    plt.close()


    print(

        "Saved:",

        output_path

    )


# ==========================================================
# PROCESS CATEGORY
# ==========================================================

def process_category(

    category,

    total_samples=10

):


    print()

    print("#" * 70)

    print(

        "PROCESSING CATEGORY:",

        category.replace(
            "_",
            " "
        ).upper()

    )

    print("#" * 70)


    for index in range(

        1,

        total_samples + 1

    ):


        create_comparison(

            category,

            index

        )


# ==========================================================
# MAIN FUNCTION
# ==========================================================

def main():

    print()

    print("=" * 70)

    print(

        "XAI QUALITATIVE EVALUATION"

    )

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

        "Output Directory:",

        OUTPUT_DIR

    )


    print()


    # ======================================================
    # CHECK XAI DIRECTORIES
    # ======================================================

    print(

        "Checking XAI Directories..."

    )


    print()


    for method_name, method_path in METHODS.items():

        if os.path.exists(

            method_path

        ):


            print(

                f"[OK] {method_name}"

            )


        else:


            print(

                f"[MISSING] {method_name}"

            )


            print(

                method_path

            )


    # ======================================================
    # PROCESS ALL CATEGORIES
    # ======================================================

    for category in CATEGORIES:


        process_category(

            category,

            total_samples=10

        )


    # ======================================================
    # COMPLETED
    # ======================================================

    print()

    print("=" * 70)

    print(

        "QUALITATIVE XAI EVALUATION COMPLETED SUCCESSFULLY!"

    )

    print("=" * 70)


    print()

    print(

        "Results Saved In:"

    )

    print(

        OUTPUT_DIR

    )


    print()

    print(

        "Generated Comparisons:"

    )


    print(

        "4 Categories × 10 Samples = 40 Comparison Figures"

    )


# ==========================================================
# RUN SCRIPT
# ==========================================================

if __name__ == "__main__":

    main()
