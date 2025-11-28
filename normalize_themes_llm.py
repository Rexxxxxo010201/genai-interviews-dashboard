import os
import json
import ast
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


# ---------------- CONFIG ----------------

load_dotenv()
API_KEY = os.getenv("GENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("GENAI_API_KEY not found in .env")

client = OpenAI(api_key=API_KEY)

MODEL = "gpt-4.1"   # update if you want to use 5.1

STUDENT_FILE = Path("student_coded.csv")
TUTOR_FILE = Path("tutor_coded.csv")

OUT_HIERARCHY = Path("theme_hierarchy.csv")
OUT_PARTICIPANT_THEMES = Path("participant_themes_normalized.csv")


# ---------------- HELPERS ----------------

def parse_theme_list(x):
    if isinstance(x, list):
        return x
    if not isinstance(x, str):
        return []
    try:
        return ast.literal_eval(x)
    except Exception:
        return []


# ---------------- MAIN ----------------

def main():

    print("Loading coded files...")
    students = pd.read_csv(STUDENT_FILE)
    tutors = pd.read_csv(TUTOR_FILE)

    coded = pd.concat([students, tutors], ignore_index=True)

    # --------- Extract ALL micro themes from both files ---------

    all_micro_themes = []

    for themes in coded["key_themes"].dropna().tolist():
        micro = parse_theme_list(themes)
        for t in micro:
            t_clean = str(t).strip()
            if t_clean:
                all_micro_themes.append(t_clean)

    all_micro_themes = sorted(list(set(all_micro_themes)))

    print(f"Found {len(all_micro_themes)} unique micro-themes.")

    # --------- LLM: Cluster into macro themes ---------

    theme_list_str = "\n".join(f"- {t}" for t in all_micro_themes)

    prompt = f"""
You are analyzing qualitative interview themes about generative AI usage by students and tutors.

Here is a list of detailed (micro) themes extracted from coded interviews:

{theme_list_str}

Your tasks:

1. Group these micro-themes into **8 to 12 macro-themes**.
2. For each macro-theme:
   - give a short, human-readable label
   - give a 1–2 sentence description
3. For each micro-theme, assign it to exactly one macro-theme.

Return STRICT JSON in this format:

{{
  "macro_themes": [
    {{
      "macro_key": "short_machine_key_like_accuracy_evaluation",
      "macro_label": "Human readable label",
      "macro_description": "1–2 sentences describing the meaning",
      "micro_themes": ["micro theme 1", "micro theme 2", ...]
    }},
    ...
  ]
}}
"""

    print("Clustering micro-themes using LLM...")
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": "You are an expert thematic analyst."},
            {"role": "user", "content": prompt},
        ],
    )

    content = resp.choices[0].message.content
    data = json.loads(content)

    macro_list = data["macro_themes"]

    # --------- Build theme_hierarchy.csv ---------

    rows = []
    for macro in macro_list:
        macro_key = macro["macro_key"]
        macro_label = macro["macro_label"]
        macro_desc = macro["macro_description"]
        for micro in macro["micro_themes"]:
            rows.append({
                "micro_theme": micro,
                "macro_theme": macro_key,
                "macro_label": macro_label,
                "macro_description": macro_desc,
            })

    hierarchy_df = pd.DataFrame(rows)
    hierarchy_df.to_csv(OUT_HIERARCHY, index=False)
    print(f"Saved theme hierarchy to {OUT_HIERARCHY}")

    # --------- Build participant_themes_normalized.csv ---------

    participant_rows = []

    for _, row in coded.iterrows():
        pid = row["participant_id"]
        role = row["role"]
        micro_list = parse_theme_list(row["key_themes"])

        for micro in micro_list:
            micro_clean = str(micro).strip()
            if not micro_clean:
                continue

            # find macro theme for this micro
            match = hierarchy_df[hierarchy_df["micro_theme"] == micro_clean]
            if match.empty:
                continue

            macro_key = match.iloc[0]["macro_theme"]

            participant_rows.append({
                "participant_id": pid,
                "role": role,
                "micro_theme": micro_clean,
                "macro_theme": macro_key,
            })

    participant_df = pd.DataFrame(participant_rows)
    participant_df.to_csv(OUT_PARTICIPANT_THEMES, index=False)
    print(f"Saved normalized participant themes to {OUT_PARTICIPANT_THEMES}")


if __name__ == "__main__":
    main()
