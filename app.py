import ast
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import numpy as np
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components



# ---------- Config ----------

DATA_DIR = Path(".")

STUDENT_CODED = DATA_DIR / "student_coded.csv"
TUTOR_CODED = DATA_DIR / "tutor_coded.csv"
STUDENT_SANKEY = DATA_DIR / "student_sankey_links.csv"
TUTOR_SANKEY = DATA_DIR / "tutor_sankey_links.csv"
TURNS_CLEAN = DATA_DIR / "interview_turns_final.csv"
THEME_QUOTES = DATA_DIR / "theme_quotes.csv"

PART_THEMES = DATA_DIR / "participant_themes_normalized.csv"
THEME_HIERARCHY = DATA_DIR / "theme_hierarchy.csv"


# ---------- Data loaders (cached) ----------

@st.cache_data
def load_coded():
    students = pd.read_csv(STUDENT_CODED)
    tutors = pd.read_csv(TUTOR_CODED)
    return students, tutors


@st.cache_data
def load_sankey_links():
    student_links = pd.read_csv(STUDENT_SANKEY)
    tutor_links = pd.read_csv(TUTOR_SANKEY)
    return student_links, tutor_links

@st.cache_data
def load_theme_quotes():
    return pd.read_csv(THEME_QUOTES)

@st.cache_data
def load_turns():
    return pd.read_csv(TURNS_CLEAN)

@st.cache_data
def load_participant_themes():
    return pd.read_csv(PART_THEMES)


@st.cache_data
def load_theme_hierarchy():
    return pd.read_csv(THEME_HIERARCHY)



# ---------- Helpers: Sankey ----------

def build_sankey_figure(links_df: pd.DataFrame, title: str):
    """
    Build a Sankey figure from a long-format link table with columns:
    source, target, value, stage_from, stage_to
    """
    if links_df.empty:
        return go.Figure()

    labels = pd.unique(pd.concat([links_df["source"], links_df["target"]], ignore_index=True))
    label_to_id = {label: i for i, label in enumerate(labels)}

    sources = links_df["source"].map(label_to_id).tolist()
    targets = links_df["target"].map(label_to_id).tolist()
    values = links_df["value"].tolist()

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=20,               # more space around labels
                    thickness=24,         # slightly thicker nodes
                    label=labels.tolist(),
                    color="gray",         # neutral node color
                    line=dict(color="white", width=0.5)
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    opacity=0.5           # softens overlapping links
                ),
            )
        ]
    )

    fig.update_layout(
        title_text=title,
        font=dict(
            family="Arial",   # ★ Best readability
            size=16,          # Increased size
            color="white",    # High contrast
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=60, b=20)
    )

    return fig


# ---------- Helpers: Themes & quotes ----------

def explode_themes(df: pd.DataFrame) -> pd.DataFrame:
    """
    key_themes is stored as a string representation of a list.
    This converts it to one row per theme.
    """
    df = df.copy()

    def parse_list(x):
        try:
            return ast.literal_eval(x)
        except Exception:
            return []

    df["theme_list"] = df["key_themes"].apply(parse_list)
    exploded = df.explode("theme_list")
    exploded = exploded.dropna(subset=["theme_list"])
    exploded = exploded.rename(columns={"theme_list": "theme"})
    return exploded


def get_quotes_for_theme(
    turns: pd.DataFrame,
    coded: pd.DataFrame,
    role_filter: str,
    theme: str,
    min_len: int = 80,
    max_quotes: int = 20,
) -> pd.DataFrame:
    """
    Return a curated set of quotes for a given role + theme.

    - role_filter: "student" or "tutor"
    - theme: theme string from key_themes
    """
    # Filter coded participants by role and theme
    coded_role = coded[coded["role"] == role_filter].copy()
    exploded = explode_themes(coded_role)
    participants_with_theme = exploded[exploded["theme"] == theme]["participant_id"].unique()

    if len(participants_with_theme) == 0:
        return pd.DataFrame(columns=["participant_id", "text", "interview_id", "question_label_propagated"])

    # Filter turns: only those participants, only participant speech, not interviewer
    df = turns.copy()
    df = df[df["participant_role"] == role_filter]
    df = df[df["participant_id"].isin(participants_with_theme)]
    df = df[df["speaker"] == df["participant_id"]]  # only their own speech

    # Filter by length
    df["text_len"] = df["text"].str.len()
    df = df[df["text_len"] >= min_len]

    if df.empty:
        return pd.DataFrame(columns=["participant_id", "text", "interview_id", "question_label_propagated"])

    # Sort so that longer quotes per participant come first
    df = df.sort_values(["participant_id", "text_len"], ascending=[True, False])

    # Limit quotes per participant to avoid spam
    df = df.groupby("participant_id").head(3)

    # Overall cap
    df = df.head(max_quotes)

    # Keep only necessary columns
    return df[["participant_id", "interview_id", "question_label_propagated", "text"]]




def build_macro_cooccurrence_matrix(part_themes: pd.DataFrame, role_filter: str | None = None):
    """
    Build a macro-theme co-occurrence matrix:
    count how many participants mention each pair of macro themes.
    """
    df = part_themes.copy()
    if role_filter is not None:
        df = df[df["role"] == role_filter]

    # one row per participant_id → set of macro themes
    grouped = df.groupby("participant_id")["macro_theme"].apply(set)

    macros = sorted(df["macro_theme"].unique())
    idx = {m: i for i, m in enumerate(macros)}
    mat = np.zeros((len(macros), len(macros)), dtype=int)

    for themes in grouped:
        themes = list(themes)
        for i in range(len(themes)):
            for j in range(i, len(themes)):
                a, b = themes[i], themes[j]
                ia, ib = idx[a], idx[b]
                mat[ia, ib] += 1
                if ia != ib:
                    mat[ib, ia] += 1

    co_df = pd.DataFrame(mat, index=macros, columns=macros)
    return co_df

def plot_macro_cooccurrence_heatmap(co_df: pd.DataFrame, title: str):
    if co_df.empty:
        return go.Figure()

    zmin = 0
    zmax = co_df.values.max()

    fig = go.Figure(
        data=go.Heatmap(
            z=co_df.values,
            x=co_df.columns,
            y=co_df.index,
            zmin=zmin,
            zmax=zmax,
            colorscale="Viridis",     # or "Blues"
            colorbar=dict(
                title="Co-occurrence",
                titleside="right",
                tickfont=dict(color="white"),
                titlefont=dict(color="white"),
            )
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Macro theme",
        yaxis_title="Macro theme",
        font=dict(color="white"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=60, r=60, t=60, b=60)
    )

    return fig



def build_theme_sunburst(hierarchy: pd.DataFrame, part_themes: pd.DataFrame, role_filter: str | None = None):
    """
    Build a sunburst where:
    - Level 1: macro_label
    - Level 2: micro_theme
    value = number of participants with that micro_theme
    """
    df = part_themes.copy()
    if role_filter is not None:
        df = df[df["role"] == role_filter]

    # count participants per micro_theme
    counts = (
        df.groupby(["micro_theme"])["participant_id"]
        .nunique()
        .reset_index(name="participants")
    )

    merged = counts.merge(hierarchy, on="micro_theme", how="left")

    if merged.empty:
        return go.Figure()

    labels = []
    parents = []
    values = []

    # Macro nodes
    macro_groups = merged.groupby("macro_theme")["participants"].sum().reset_index()
    macro_label_map = (
        hierarchy[["macro_theme", "macro_label"]]
        .drop_duplicates()
        .set_index("macro_theme")["macro_label"]
        .to_dict()
    )

    for _, row in macro_groups.iterrows():
        mkey = row["macro_theme"]
        mlabel = macro_label_map.get(mkey, mkey)
        labels.append(mlabel)
        parents.append("")
        values.append(row["participants"])

    # Micro nodes
    for _, row in merged.iterrows():
        mkey = row["macro_theme"]
        mlabel = macro_label_map.get(mkey, mkey)
        micro = row["micro_theme"]
        labels.append(micro)
        parents.append(mlabel)
        values.append(row["participants"])

    fig = go.Figure(
        go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
        )
    )
    fig.update_layout(
        margin=dict(t=40, l=0, r=0, b=0),
        title="Macro and micro theme structure",
    )
    return fig



def compute_macro_counts(part_themes: pd.DataFrame, role_filter: str | None = None):
    df = part_themes.copy()
    if role_filter is not None:
        df = df[df["role"] == role_filter]

    counts = (
        df.groupby("macro_theme")["participant_id"]
        .nunique()
        .reset_index(name="participants")
    )
    return counts


def compute_micro_counts_for_macro(
    part_themes: pd.DataFrame,
    hierarchy: pd.DataFrame,
    macro_theme: str,
    role_filter: str | None = None,
):
    df = part_themes.copy()
    if role_filter is not None:
        df = df[df["role"] == role_filter]

    df = df[df["macro_theme"] == macro_theme]

    if df.empty:
        return pd.DataFrame(columns=["micro_theme", "participants"])

    counts = (
        df.groupby("micro_theme")["participant_id"]
        .nunique()
        .reset_index(name="participants")
    )

   
    return counts.sort_values("participants", ascending=False)

def get_macro_label_map(hierarchy: pd.DataFrame) -> dict:
    return (
        hierarchy[["macro_theme", "macro_label"]]
        .drop_duplicates()
        .set_index("macro_theme")["macro_label"]
        .to_dict()
    )




# ---------- Pages ----------

def overview_page(students, tutors, student_links, tutor_links):
    st.subheader("Sankey pathways")

    who = st.radio("View", ["Students", "Tutors"], horizontal=True)

    if who == "Students":
        st.markdown("### Student journey: Familiarity → Usage → Use case → Impact")

        fig = build_sankey_figure(
            student_links,
            title="Student GenAI pathways",
        )
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Students (coded)", len(students[students["role"] == "student"]))
        with c2:
            st.metric(
                "High familiarity",
                (students["familiarity_level"] == "high").sum(),
            )
        with c3:
            st.metric(
                "Heavy users",
                (students["usage_frequency"] == "heavy").sum(),
            )

    else:
        st.markdown("### Tutor journey: Familiarity → Integration → Impact → Ethics concern")

        fig = build_sankey_figure(
            tutor_links,
            title="Tutor GenAI pathways",
        )
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Tutors (coded)", len(tutors[tutors["role"] == "tutor"]))
        with c2:
            st.metric(
                "High familiarity",
                (tutors["familiarity_level"] == "high").sum(),
            )
        with c3:
            st.metric(
                "Central integration",
                (tutors["integration_depth"] == "central to assessment").sum(),
            )


def key_themes_and_quotes_page(students, tutors, turns, theme_quotes):
    st.subheader("Key themes & quotes")

    role_choice = st.radio("Role", ["Students", "Tutors"], horizontal=True)
    if role_choice == "Students":
        coded = students
        role_filter = "student"
        st.markdown("### Themes in student experiences with GenAI")
    else:
        coded = tutors
        role_filter = "tutor"
        st.markdown("### Themes in tutor experiences with GenAI")

    # ---- THEME FREQUENCY (from coded data) ----
    coded_role = coded[coded["role"] == role_filter].copy()
    exploded = explode_themes(coded_role)

    if exploded.empty:
        st.info("No themes available for this role.")
        return

    theme_counts = (
        exploded.groupby("theme")
        .size()
        .reset_index(name="participants")
        .sort_values("participants", ascending=False)
    )

    st.markdown("#### Theme frequency")
    st.caption("Number of participants whose coded summary includes each theme.")
    st.dataframe(theme_counts, use_container_width=True, height=300)

    st.markdown("#### Bar chart of top themes")
    top_n = st.slider("How many top themes to show?", 5, min(20, len(theme_counts)), 10)
    top = theme_counts.head(top_n).set_index("theme")
    st.bar_chart(top)

    # ---- QUOTES (from theme_quotes.csv) ----
    st.markdown("### Quotes for a selected theme")

    theme_options = theme_counts["theme"].tolist()
    selected_theme = st.selectbox("Select a theme to view quotes", theme_options)

    # Filter quotes for this role + theme
    qdf = theme_quotes[
        (theme_quotes["role"] == role_filter) &
        (theme_quotes["theme"] == selected_theme)
    ].copy()

    if qdf.empty:
        st.info("No short verbatim quotes available for this theme (yet).")
        return

    st.caption(
        "Showing short, verbatim quotes extracted for this theme. "
        "Each quote is a real phrase from the transcript."
    )

    for _, row in qdf.iterrows():
        st.markdown("---")
        meta = f"**{role_filter.title()}** • `{row['participant_id']}` • `{row['theme']}`"
        st.markdown(meta)
        st.markdown(f"> {row['quote']}")




def theme_structure_page(part_themes, hierarchy):
    st.subheader("Theme structure")

    role_choice = st.radio("Role filter", ["All", "Students", "Tutors"], horizontal=True)
    if role_choice == "All":
        role_filter = None
    elif role_choice == "Students":
        role_filter = "student"
    else:
        role_filter = "tutor"

    # --- 1. Macro co-occurrence heatmap ---

    st.markdown("### Macro theme co-occurrence")
    st.caption("Shows how often macro themes co-occur within the same participant.")

    co_df = build_macro_cooccurrence_matrix(part_themes, role_filter)
    if co_df.empty:
        st.info("No data available for co-occurrence.")
    else:
        fig_heat = plot_macro_cooccurrence_heatmap(co_df, "Macro theme co-occurrence")
        st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")

  

        # --- 2. Macro and micro theme structure (simplified) ---

    st.markdown("### Macro and micro theme hierarchy")

    macro_label_map = get_macro_label_map(hierarchy)

    # 2a. Macro theme bar chart
    st.caption("Number of participants whose coded summaries include each macro theme.")
    macro_counts = compute_macro_counts(part_themes, role_filter)
    if macro_counts.empty:
        st.info("No macro theme data available.")
    else:
        macro_counts = macro_counts.sort_values("participants", ascending=False)
        macro_counts["macro_label"] = macro_counts["macro_theme"].map(
            lambda k: macro_label_map.get(k, k)
        )

        st.bar_chart(
            macro_counts.set_index("macro_label")["participants"],
            use_container_width=True,
        )

        # 2b. Micro themes for a selected macro
        st.markdown("#### Micro themes under a selected macro theme")

        options = macro_counts["macro_theme"].tolist()
        default_macro = options[0] if options else None
        selected_macro = st.selectbox(
            "Choose a macro theme",
            options,
            format_func=lambda k: macro_label_map.get(k, k),
            index=0 if default_macro is not None else None,
        )

        if selected_macro is not None:
            micro_counts = compute_micro_counts_for_macro(
                part_themes, hierarchy, selected_macro, role_filter
            )
            if micro_counts.empty:
                st.info("No micro themes found for this macro theme.")
            else:
                st.caption("Participants per micro theme under this macro theme.")
                st.bar_chart(
                    micro_counts.set_index("micro_theme")["participants"],
                    use_container_width=True,
                )



# ---------- Main ----------

def main():
    st.set_page_config(
        page_title="GenAI Learning Insights Dashboard",
        layout="wide",
    )

    st.title("GenAI Learning Insights Dashboard")
    st.caption("Students and tutors' perspectives on generative AI in 31266 / 31269")

  

    page = st.sidebar.radio(
    "Navigate",
    ["Sankey pathways", "Key themes & quotes", "Theme structure"],
)


    students, tutors = load_coded()
    student_links, tutor_links = load_sankey_links()
    turns = load_turns()
    theme_quotes = load_theme_quotes()  # NEW


    
    part_themes = load_participant_themes()
    hierarchy = load_theme_hierarchy()



    if page == "Sankey pathways":
        overview_page(students, tutors, student_links, tutor_links)
    elif page == "Key themes & quotes":
        key_themes_and_quotes_page(students, tutors, turns, theme_quotes)
    elif page == "Theme structure":
        theme_structure_page(part_themes, hierarchy)


if __name__ == "__main__":
    main()
