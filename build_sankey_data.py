import pandas as pd
from pathlib import Path

STUDENT_CODED = Path("student_coded.csv")
TUTOR_CODED = Path("tutor_coded.csv")

STUDENT_SANKEY_OUT = Path("student_sankey_links.csv")
TUTOR_SANKEY_OUT = Path("tutor_sankey_links.csv")


def build_links(df: pd.DataFrame, stages, role_filter: str) -> pd.DataFrame:
    """
    Build a long-format link table for a multi-step Sankey.

    stages: ordered list of column names, e.g.
        ["familiarity_level", "usage_frequency", "primary_use_case",
         "perceived_impact_on_learning"]

    Returns columns:
        source, target, value, stage_from, stage_to
    """
    df_role = df[df["role"] == role_filter].copy()

    # Drop rows with missing values in any of the required stages
    df_role = df_role.dropna(subset=stages)

    links = []
    for i in range(len(stages) - 1):
        s_col, t_col = stages[i], stages[i + 1]

        tmp = (
            df_role
            .groupby([s_col, t_col])
            .size()
            .reset_index(name="value")
            .rename(columns={s_col: "source", t_col: "target"})
        )
        tmp["stage_from"] = s_col
        tmp["stage_to"] = t_col
        links.append(tmp)

    if links:
        return pd.concat(links, ignore_index=True)
    else:
        return pd.DataFrame(columns=["source", "target", "value", "stage_from", "stage_to"])


def main():
    student = pd.read_csv(STUDENT_CODED)
    tutor = pd.read_csv(TUTOR_CODED)

    # --- Student Sankey: Familiarity → Usage → Use-case → Impact ---
    student_stages = [
        "familiarity_level",
        "usage_frequency",
        "primary_use_case",
        "perceived_impact_on_learning",
    ]

    student_links = build_links(student, student_stages, role_filter="student")
    student_links.to_csv(STUDENT_SANKEY_OUT, index=False)
    print(f"Saved student Sankey links to {STUDENT_SANKEY_OUT}")

    # --- Tutor Sankey: Familiarity → Integration → Impact → Ethics concern ---
    tutor_stages = [
        "familiarity_level",
        "integration_depth",
        "perception_of_learning_impact",
        "primary_ethics_concern",
    ]

    tutor_links = build_links(tutor, tutor_stages, role_filter="tutor")
    tutor_links.to_csv(TUTOR_SANKEY_OUT, index=False)
    print(f"Saved tutor Sankey links to {TUTOR_SANKEY_OUT}")


if __name__ == "__main__":
    main()
