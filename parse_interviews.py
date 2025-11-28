import re
from pathlib import Path

import docx
import pandas as pd

STUDENT_DOC = Path("ak_Combined Student Interviews 31266_31269 copy.docx")
TUTOR_DOC   = Path("ak_Combined Tutor Interviews 31266_31269 copy.docx")

# --- helpers ---

HEADING_RE = re.compile(
    r'^(?P<subject>\d{5})\s+(?P<semester>Autmn|Autumn|Spring)\s+–\s+(?P<round>First|Second|Third)\s+(?P<type>Student|Tutor)\s+Interview',
    re.IGNORECASE
)

SPEAKER_RE = re.compile(r'^(?P<speaker>[A-Z\-]+[A-Z0-9\-]*)\s*:(?P<rest>.*)$')

def normalise_semester(s: str) -> str:
    s = s.strip().title()
    return "Autumn" if s.startswith("Autm") else s  # fix "Autmn" typo

def parse_docx(path: Path, doc_kind: str):
    """
    doc_kind: 'student' or 'tutor' (just a label for safety)
    Returns: list of dicts, one per speaker turn
    """
    doc = docx.Document(path)
    rows = []

    current_subject = None
    current_semester = None
    current_round = None
    participant_role = None
    participant_id = None

    current_speaker = None
    current_text_chunks = []
    turn_index = 0

    def flush_turn():
        nonlocal turn_index, current_speaker, current_text_chunks
        if current_speaker is None or not current_text_chunks:
            return
        turn_index += 1
        text = " ".join(t.strip() for t in current_text_chunks if t.strip())
        if not text:
            return
        rows.append({
            "doc_type": doc_kind,
            "subject_code": current_subject,
            "semester": current_semester,
            "interview_round": current_round,
            "participant_role": participant_role,
            "participant_id": participant_id,
            "speaker": current_speaker,
            "turn_index": turn_index,
            "text": text
        })
        current_text_chunks = []

    for para in doc.paragraphs:
        t = para.text.strip()
        if not t:
            continue

        # 1) Heading?
        m_head = HEADING_RE.match(t)
        if m_head:
            # new interview starts
            flush_turn()
            current_subject = m_head.group("subject")
            current_semester = normalise_semester(m_head.group("semester"))
            current_round = m_head.group("round").title()
            turn_index = 0
            # participant info will be set once we see first student/tutor speaker
            continue

        # 2) Speaker line?
        m_s = SPEAKER_RE.match(t)
        if m_s:
            speaker_raw = m_s.group("speaker")
            first_text = m_s.group("rest").strip()

            # finish previous turn
            flush_turn()

            current_speaker = "Interviewer" if speaker_raw.startswith("Interviewer") else speaker_raw

            # update participant info when we first see a participant label
            if speaker_raw.startswith(("AUT-S", "SPR-S")):
                participant_role = "student"
                participant_id = speaker_raw
            elif speaker_raw.startswith(("AUT-P", "SPR-P")):
                participant_role = "tutor"
                participant_id = speaker_raw

            current_text_chunks = []
            if first_text:
                current_text_chunks.append(first_text)
            continue

        # 3) Continuation of same speaker
        if current_speaker is not None:
            current_text_chunks.append(t)
        else:
            # text before any speaker: usually intro fluff, can ignore or log
            continue

    # flush last turn
    flush_turn()

    return rows

student_rows = parse_docx(STUDENT_DOC, doc_kind="student")
tutor_rows = parse_docx(TUTOR_DOC, doc_kind="tutor")

# df = pd.DataFrame(student_rows + tutor_rows)
# print(df.head())
# print(df["participant_id"].value_counts())

# # optionally save to CSV for inspection
# df.to_csv("interview_turns_parsed.csv", index=False)


df = pd.DataFrame(student_rows + tutor_rows)
print("Parsed rows:", len(df))
print(df.head())

df.to_csv("interview_turns_parsed.csv", index=False)
print("CSV saved!")

