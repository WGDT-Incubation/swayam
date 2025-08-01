import re
import json
import csv

def extract_course_units_from_text(text: str, course_name: str,
                                   json_out: str, csv_out: str) -> None:
    """
    Extracts topics and subtopics from a structured course text block.

    Parameters:
    - text: full text containing course units
    - course_name: e.g. "Business Statistics"
    - json_out: path to save JSON output
    - csv_out: path to save CSV output
    """
    data = {"course": course_name, "topics": []}
    csv_rows = []

    # Split into units
    unit_pattern = re.compile(r"Unit\s+\d+\s*:\s*(.+)")
    unit_matches = list(unit_pattern.finditer(text))
    
    for i, match in enumerate(unit_matches):
        topic = match.group(1).strip()

        start_idx = match.end()
        end_idx = unit_matches[i+1].start() if i+1 < len(unit_matches) else len(text)
        subtext = text[start_idx:end_idx].strip()

        # Break into lines and extract bullet or sentence style subtopics
        lines = re.split(r"(?:\([a-zA-Z]+\)|[a-zA-Z]+\))\s+|[\n•;\-]+", subtext)
        raw_subtopics = [line.strip() for line in lines if line.strip()]

        # Further split where necessary
        refined_subtopics = []
        for line in raw_subtopics:
            # Skip if line is just numbers or symbols
            if re.match(r"^[√✓\d\s]*$", line):
                continue
            parts = re.split(r"[.;]", line)
            refined_subtopics.extend([p.strip() for p in parts if p.strip()])

        # De-duplicate
        seen = set()
        cleaned = []
        for s in refined_subtopics:
            if s not in seen:
                seen.add(s)
                cleaned.append(s)

        data["topics"].append({"topic": topic, "subtopics": cleaned})
        for sub in cleaned:
            csv_rows.append({"topic": topic, "subtopic": sub})

    # Save JSON
    with open(json_out, "w", encoding="utf-8") as jf:
        json.dump(data, jf, ensure_ascii=False, indent=2)

    # Save CSV
    with open(csv_out, "w", encoding="utf-8", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=["topic", "subtopic"])
        writer.writeheader()
        writer.writerows(csv_rows)


# ----------------- Example Usage -----------------
if __name__ == "__main__":
    with open("/workspaces/swayam/result/BS/raw/ugc_business_statistics_raw.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    extract_course_units_from_text(
        text=raw_text,
        course_name="Business Statistics",
        json_out="/workspaces/swayam/result/BS/2506syllabus_business_statistics_topics.json",
        csv_out="/workspaces/swayam/result/BS/2506syllabus_business_statistics_topics.csv"
    )
