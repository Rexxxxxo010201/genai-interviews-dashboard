import pandas as pd

INPUT = "interview_turns_clean_labeled.csv"
OUTPUT = "interview_turns_merged.csv"

# threshold for "short" text that likely doesn't stand alone
SHORT_THRESHOLD = 40


def merge_adjacent_turns(df: pd.DataFrame) -> pd.DataFrame:
    merged_rows = []
    buffer = None  # holds (row dict)

    for _, row in df.iterrows():
        text = str(row["text"]).strip()

        # If buffer exists, check if this row should merge into it
        if buffer is not None:
            same_speaker = row["speaker"] == buffer["speaker"]
            same_interview = row["interview_id"] == buffer["interview_id"]
            same_label = row["question_label_propagated"] == buffer["question_label_propagated"]

            if (
                same_speaker
                and same_interview
                and same_label
                and (len(text) < SHORT_THRESHOLD or len(buffer["text"]) < SHORT_THRESHOLD)
            ):
                # merge texts
                buffer["text"] = (buffer["text"] + " " + text).strip()
                continue
            else:
                merged_rows.append(buffer)
                buffer = row.to_dict()
        else:
            buffer = row.to_dict()

    # flush final buffer
    if buffer is not None:
        merged_rows.append(buffer)

    return pd.DataFrame(merged_rows)


def main():
    df = pd.read_csv(INPUT)

    # sort by interview + turn_index to preserve order
    df = df.sort_values(["interview_id", "turn_index"]).reset_index(drop=True)

    print("Merging short adjacent turns...")
    merged_df = merge_adjacent_turns(df)

    # Recompute turn_index after merging
    merged_df["turn_index"] = (
        merged_df.groupby("interview_id").cumcount() + 1
    )

    print(f"Saving merged dataset to {OUTPUT}")
    merged_df.to_csv(OUTPUT, index=False)

    print("Done. Please use interview_turns_merged.csv for next steps.")


if __name__ == "__main__":
    main()
