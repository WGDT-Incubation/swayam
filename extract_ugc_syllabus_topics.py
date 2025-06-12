
"""extract_ugc_syllabus_topics.py
----------------------------------
Extracts topics and sub‑topics from the UGC LOCF Financial‑Accounting syllabus
PDF (UGC_8737712_LOCF-document-on-commerce-2.pdf).

Output:
  1. JSON  – hierarchical structure: course → topics → subtopics
  2. CSV   – flat table with columns: topic, subtopic
"""

import fitz          # PyMuPDF
import re
import json
import csv


def extract_ugc_financial_accounting(pdf_path: str,
                                     json_out: str,
                                     csv_out: str) -> None:
    """Parse the specified UGC PDF and create JSON + CSV outputs."""

    data = {"course": "Financial Accounting", "topics": []}
    csv_rows = []

    # ------------------------------------------------------------
    # Read text from the PDF – start capturing after “COURSE CONTENTS”
    # ------------------------------------------------------------
    doc = fitz.open(pdf_path)
    full_text = []
    capture = False

    for page in doc:
        page_text = page.get_text()
        if "COURSE CONTENTS" in page_text:
            capture = True
        if capture:
            full_text.append(page_text)

    full_text = "\n".join(full_text)

    # ------------------------------------------------------------
    # Keep everything starting from the first “Unit n:”
    # ------------------------------------------------------------
    start_match = re.search(r"Unit\s+\d+\s*:", full_text)
    if not start_match:
        raise ValueError("No 'Unit n:' header found in the PDF text.")
    content = full_text[start_match.start():]

    # ------------------------------------------------------------
    # Split content into blocks:  Unit n:  <body>
    # ------------------------------------------------------------
    parts = re.split(r"(Unit\s+\d+\s*:)", content)
    units = ["".join(parts[i:i + 2]) for i in range(1, len(parts), 2)]

    for unit in units:
        # ---------- Extract topic ----------
        topic_match = re.match(
            r"Unit\s+\d+\s*:\s*(?:\([a-zA-Z]+\)\s*)?([^
]+)", unit)
        if not topic_match:
            continue

        topic = topic_match.group(1).strip()

        # ---------- Extract subtopics ----------
        body_text = unit[topic_match.end():].strip()

        # (i) / (ii) style roman bullets
        bullet_matches = re.findall(
            r"\(i+\)[.)]?\s*(.*?)(?=\(i+\)|$)",
            body_text,
            flags=re.IGNORECASE | re.DOTALL)

        if bullet_matches:
            candidate_lines = bullet_matches
        else:
            # Fallback: split on newline, bullets, dashes etc.
            candidate_lines = re.split(r"[\n•\-]+", body_text)

        subtopics = []
        for line in candidate_lines:
            # Further split on punctuation to get shorter sub‑topics
            pieces = re.split(r"[.;]", line)
            subtopics.extend([p.strip() for p in pieces if p.strip()])

        # De‑duplicate while preserving order
        seen = set()
        cleaned_subtopics = []
        for s in subtopics:
            if s not in seen:
                seen.add(s)
                cleaned_subtopics.append(s)

        # ---------- Store ----------
        data["topics"].append({"topic": topic,
                              "subtopics": cleaned_subtopics})

        for sub in cleaned_subtopics:
            csv_rows.append({"topic": topic, "subtopic": sub})

    # ------------------------------------------------------------
    # Write JSON
    # ------------------------------------------------------------
    with open(json_out, "w", encoding="utf-8") as jf:
        json.dump(data, jf, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------
    # Write CSV
    # ------------------------------------------------------------
    with open(csv_out, "w", encoding="utf-8", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=["topic", "subtopic"])
        writer.writeheader()
        writer.writerows(csv_rows)


# -----------------------------------------------------------------
# Example usage – uncomment and edit the paths as required
# -----------------------------------------------------------------
# extract_ugc_financial_accounting(
#     "UGC_8737712_LOCF-document-on-commerce-2.pdf",
#     "ugc_syllabus.json",
#     "ugc_syllabus.csv"
# )
