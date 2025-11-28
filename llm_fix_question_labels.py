import os
import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# ------------- Config -------------

load_dotenv()
API_KEY = os.getenv("GENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("GENAI_API_KEY not found in .env")

client = OpenAI(api_key=API_KEY)

MODEL_NAME = "gpt-4.1"  # change to your 5.1 model name when available

INPUT_CSV = Path("interview_turns_clean_labeled.csv")
OUTPUT_CSV = Path("interview_turns_clean_labeled.csv")  # overwrite in-place
# If you want a backup, use a different name first.


# ------------- Prompt builders -------------

STUDENT_LABELS = [
    "Q_USAGE",
    "Q_ASSESSMENT_HELPFULNESS",
    "Q_CONFIDENCE",
    "Q_CRITICAL_EVAL",
    "Q_ERROR_HANDLING",
    "Q_ETHICS",
    "Q_POLICY",
    "Q_SUPPORT",
    "Q_FUTURE_ROLE",
    "Q_BACKGROUND",
    "Q_OTHER",
]

TUTOR_LABELS = [
    "Q_FAMILIARITY",
    "Q_USAGE_FREQ",
    "Q_INTEGRATION",
    "Q_STUDENT_IMPACT",
    "Q_CRITICAL_EVAL",
    "Q_ETHICS",
    "Q_GUIDELINES_SCAFFOLDING",
    "Q_FUTURE_IMPROVEMENTS",
    "Q_SUPPORT",
    "Q_BACKGROUND",
    "Q_OTHER",
]


def build_question_prompt(text: str, doc_type: str) -> str:
    if doc_type == "student":
        labels = STUDENT_LABELS
        context = "This is an interviewer question asked to a student about their use of generative AI in a subject."
    else:
        labels = TUTOR_LABELS
        context = "This is an interviewer question asked to a tutor about their experience with generative AI in teaching."

    labels_str = ", ".join(labels)

    return f"""
You are classifying an interviewer utterance from a semi-structured interview about generative AI in higher education.

{context}

Question text (or interviewer utterance):

\"\"\"{text.strip()}\"\"\"

Your task: choose exactly ONE label from the following set that best describes the PRIMARY purpose of this interviewer utterance:

{labels_str}

Rules:
- If it is clearly about how the person USED or USES GenAI, choose a usage-related label.
- If it is about ethics, right/wrong use, or dilemmas, choose an ethics-related label.
- If it is just introductory background or small talk, choose Q_BACKGROUND.
- If it does not fit any label well, choose Q_OTHER.
- Always choose one label, even if you are uncertain.

Return ONLY valid JSON of the form:
{{ "label": "<ONE_OF_THE_LABELS_ABOVE>" }}
"""


def call_llm_for_label(question_text: str, doc_type: str) -> str:
    prompt = build_question_prompt(question_text, doc_type)
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert qualitative coder. "
                    "Always respond with STRICT JSON, no explanation, no markdown."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    content = resp.choices[0].message.content
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
        data = json.loads(cleaned)
    label = data.get("label")
    return label


# ------------- Propagation helper -------------

def propagate_labels(df: pd.DataFrame, label_col: str, out_col: str) -> pd.DataFrame:
    """
    For each interview_id, walk through turns.
    When we see an interviewer with a non-null label_col,
    remember it and assign to subsequent turns until the next labelled interviewer.
    """
    groups = []
    for iid, g in df.groupby("interview_id", sort=True):
        g = g.sort_values("turn_index").copy()
        current_label = None
        propagated = []
        for _, r in g.iterrows():
            if r["speaker"] == "Interviewer" and pd.notna(r[label_col]):
                current_label = r[label_col]
                propagated.append(current_label)
            else:
                propagated.append(current_label)
        g[out_col] = propagated
        groups.append(g)
    return pd.concat(groups, ignore_index=True)


# ------------- Main -------------

def main():
    df = pd.read_csv(INPUT_CSV)

    # We will create a new column question_label_llm, then overwrite question_label & question_label_propagated.
    df["question_label_llm"] = df.get("question_label", pd.Series([None]*len(df)))

    mask = df["speaker"] == "Interviewer"

    # Only relabel interviewer rows
    for idx in df[mask].index:
        row = df.loc[idx]
        text = str(row["text"]).strip()
        doc_type = row["doc_type"]

        if not text:
            continue

        print(f"Classifying interviewer turn {idx} (doc_type={doc_type})...")
        label = call_llm_for_label(text, doc_type)
        df.at[idx, "question_label_llm"] = label

    # Now propagate using the LLM labels
    df = propagate_labels(df, label_col="question_label_llm", out_col="question_label_propagated_llm")

    # Option A: overwrite original columns so the app uses the fixed ones without changes
    df["question_label"] = df["question_label_llm"]
    df["question_label_propagated"] = df["question_label_propagated_llm"]

    # Optionally drop the helper columns if you want it clean:
    df = df.drop(columns=["question_label_llm", "question_label_propagated_llm"], errors="ignore")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Updated question labels saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
