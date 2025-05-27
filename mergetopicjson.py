import pandas as pd
import json
import re
from collections import defaultdict

# --- Paths ---
MATCHED_CSV = "/workspaces/swayam/result/correct_matched_topics_subtopics_comparison.csv"
MATCHED_JSON_1 = "/workspaces/swayam/result/2705final_grouped_by_syllabus.json"
MATCHED_JSON_2 = "/workspaces/swayam/result/2705final_subtopics_by_merged_lecture.json"

# --- Load final correct CSV ---
df = pd.read_csv(MATCHED_CSV)

def clean(text):
    if pd.isna(text):
        return ""
    return str(text).strip()

# --- JSON 1: Matched Syllabus Topic → Lecture Topic(s) → Subtopics ---
def build_json_grouped_by_syllabus():
    structure = {"course": "Financial Accounting", "topics": []}
    syllabus_map = defaultdict(lambda: defaultdict(set))

    for _, row in df.iterrows():
        st = clean(row["Matched Syllabus Topic"])
        lt = clean(row["Lecture Topic"])
        sub = clean(row["Lecture Subtopic"])

        if not st or not lt:
            continue

        # Split subtopics by comma or semicolon
        for s in re.split(r"[;,]", sub):
            sub_clean = s.strip()
            if sub_clean:
                syllabus_map[st][lt].add(sub_clean)

    for st, ltopics in syllabus_map.items():
        children = []
        for lt, subs in ltopics.items():
            children.append({"lecture_topic": lt, "subtopics": sorted(subs)})
        structure["topics"].append({"topic": st, "children": children})

    with open(MATCHED_JSON_1, "w") as f:
        json.dump(structure, f, indent=2)
    print(f"✅ Saved grouped JSON to {MATCHED_JSON_1}")

# --- JSON 2: Matched Syllabus Topic → Merged Lecture Topic Name → Unique Subtopics ---
def build_json_merged_subtopics():
    structure = {"course": "Financial Accounting", "topics": []}
    topic_map = defaultdict(lambda: defaultdict(set))

    for _, row in df.iterrows():
        st = clean(row["Matched Syllabus Topic"])
        lt = clean(row["Lecture Topic"])
        sub = clean(row["Lecture Subtopic"])

        if not st or not lt:
            continue

        for s in re.split(r"[;,]", sub):
            sub_clean = s.strip()
            if sub_clean:
                topic_map[st][lt].add(sub_clean)

    for st, lectures in topic_map.items():
        all_subs = set()
        for l, subs in lectures.items():
            all_subs.update(subs)
        merged_name = " - ".join(sorted(lectures.keys()))
        structure["topics"].append({"topic": st, "lecture_group": merged_name, "subtopics": sorted(all_subs)})

    with open(MATCHED_JSON_2, "w") as f:
        json.dump(structure, f, indent=2)
    print(f"✅ Saved merged JSON to {MATCHED_JSON_2}")

if __name__ == "__main__":
    build_json_grouped_by_syllabus()
    build_json_merged_subtopics()
