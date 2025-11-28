import os
import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# -------------------------
# Config
# -------------------------

load_dotenv()  # loads .env at project root
OPENAI_API_KEY = os.getenv("GENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("GENAI_API_KEY not found in .env")

client = OpenAI(api_key=OPENAI_API_KEY)

# You can change this to whatever model your account supports.
# If you have access to a 5.1-style model, e.g. "gpt-5.1", put that here.
MODEL_NAME = "gpt-5.1"

INPUT_CSV = Path("interview_turns_clean_labeled.csv")
STUDENT_OUT = Path("student_coded.csv")
TUTOR_OUT = Path("tutor_coded.csv")

# -------------------------
# Load data
# -------------------------

df = pd.read_csv(INPUT_CSV)

# We only want participant turns (exclude interviewer-only turns)
df_part = df[df["participant_id"].notna()].copy()

# -------------------------
# Helper: build text per participant
# -------------------------

def build_participant_context(group: pd.DataFrame) -> str:
    """
    Build a consolidated text block for a single participant
    that preserves some structure by question.
    """
    lines = []
    # Sort by interview + turn order
    group = group.sort_values(["interview_id", "turn_index"])

    for _, row in group.iterrows():
        q_label = row.get("question_label_propagated", None)
        text = str(row["text"]).strip()
        if not text:
            continue

        if isinstance(q_label, str) and q_label:
            lines.append(f"[{q_label}] {text}")
        else:
            lines.append(text)

    return "\n".join(lines)


# -------------------------
# Prompt builders
# -------------------------

def build_student_prompt(text_block: str, participant_id: str) -> str:
    return f"""
You are analysing qualitative interview data from a university student about their use of generative AI tools in a subject.

The text below contains this student's responses across multiple questions. Each line may optionally start with a tag like [Q_USAGE], [Q_ETHICS], etc, indicating which question it relates to.

Your task: read ALL the text, then classify this student along several dimensions and extract key themes.

Text to analyse (student ID: {participant_id}):

\"\"\"{text_block}\"\"\"


Return ONLY valid JSON with the following structure and allowed values:

{{
  "role": "student",
  "summary": string,              // 2-4 sentence plain-language summary of this student's overall stance and experience
  "familiarity_level": string,    // one of: "none", "low", "medium", "high"
  "usage_frequency": string,      // one of: "non user", "light", "moderate", "heavy"
  "primary_use_case": string,     // short phrase, e.g. "draft writing", "idea generation", "checking answers", "coding help", "study planning", "did not use"
  "critical_evaluation_skill": string,  // one of: "no checking", "occasional checking", "systematic checking"
  "ethical_confidence": string,   // one of: "unclear", "somewhat clear", "very clear"
  "perceived_impact_on_learning": string, // one of: "strongly positive", "mixed", "minimal", "negative", "unclear"
  "key_themes": [string, ...]     // 3-8 short thematic labels, e.g. "speed and efficiency", "concerns about accuracy", "over-reliance risk"
}}

Important:
- Base your classifications ONLY on the provided text.
- If evidence is weak or mixed, choose the closest category and reflect uncertainty in the summary.
- Do NOT include any text before or after the JSON.
"""


def build_tutor_prompt(text_block: str, participant_id: str) -> str:
    return f"""
You are analysing qualitative interview data from a university tutor about their experience with generative AI in teaching and assessment.

The text below contains this tutor's responses across multiple questions. Each line may optionally start with a tag like [Q_INTEGRATION], [Q_ETHICS], etc, indicating which question it relates to.

Your task: read ALL the text, then classify this tutor along several dimensions and extract key themes.

Text to analyse (tutor ID: {participant_id}):

\"\"\"{text_block}\"\"\"


Return ONLY valid JSON with the following structure and allowed values:

{{
  "role": "tutor",
  "summary": string,                 // 2-4 sentence summary of this tutor's stance and experience with GenAI in their subject
  "familiarity_level": string,       // one of: "low", "medium", "high"
  "integration_depth": string,       // one of: "no integration", "limited examples", "embedded in activities", "central to assessment"
  "observed_student_use_pattern": string,  // one of: "very low", "selective", "frequent but superficial", "frequent and critical", "unclear"
  "observed_critical_evaluation": string,  // one of: "mostly uncritical", "mixed", "mostly critical", "unclear"
  "primary_ethics_concern": string,  // e.g. "plagiarism or originality", "over reliance", "accuracy or hallucinations", "equity or access", "minimal concerns"
  "perception_of_learning_impact": string, // one of: "net positive", "mixed", "unclear", "net negative"
  "key_themes": [string, ...]       // 3-8 short thematic labels e.g. "assessment integrity", "workload reduction", "student over-reliance"
}}

Important:
- Base your classifications ONLY on the provided text.
- If evidence is weak or mixed, choose the closest category and reflect nuance in the summary.
- Do NOT include any text before or after the JSON.
"""


# -------------------------
# LLM call helper
# -------------------------

def call_llm(prompt: str) -> dict:
    """Call OpenAI and return parsed JSON."""
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert qualitative researcher. "
                    "Always respond with STRICT JSON, no explanation, no markdown."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    content = resp.choices[0].message.content
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        # In case the model accidentally wraps in ```json``` or adds text,
        # try to clean it a bit.
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            # remove leading 'json' if present
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
        return json.loads(cleaned)


# -------------------------
# Main coding loop
# -------------------------

def main():
    records = []

    # Group by doc_type + participant_id
    for (doc_type, pid), group in df_part.groupby(["doc_type", "participant_id"]):
        text_block = build_participant_context(group)
        if not text_block.strip():
            continue

        if doc_type == "student":
            prompt = build_student_prompt(text_block, pid)
        else:
            prompt = build_tutor_prompt(text_block, pid)

        print(f"Coding {doc_type} {pid}...")

        result = call_llm(prompt)

        base_info = {
            "doc_type": doc_type,
            "participant_id": pid,
            "subject_codes": ";".join(sorted(set(group["subject_code"].astype(str)))),
            "semesters": ";".join(sorted(set(group["semester"].astype(str)))),
        }

        # Flatten JSON into one record
        flat = {**base_info, **result}
        records.append(flat)

    coded_df = pd.DataFrame(records)

    # Split into student / tutor outputs
    if not coded_df.empty:
        student_df = coded_df[coded_df["role"] == "student"].copy()
        tutor_df = coded_df[coded_df["role"] == "tutor"].copy()

        if not student_df.empty:
            student_df.to_csv(STUDENT_OUT, index=False)
            print(f"Saved student-coded data to {STUDENT_OUT}")

        if not tutor_df.empty:
            tutor_df.to_csv(TUTOR_OUT, index=False)
            print(f"Saved tutor-coded data to {TUTOR_OUT}")
    else:
        print("No records coded. Check input data.")


if __name__ == "__main__":
    main()
