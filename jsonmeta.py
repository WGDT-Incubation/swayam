import os
import pandas as pd
import json

# Configuration
LECTURE_TOPICS_CSV = "/workspaces/swayam/result/1205lecture_topics_output.csv"
OUTPUT_JSON = "/workspaces/swayam/result/financial_accounting_topics_hierarchy.json"
COURSE_NAME = "Financial Accounting"

# Read lecture topic + subtopics
if not os.path.exists(LECTURE_TOPICS_CSV):
    raise FileNotFoundError("Lecture topics CSV not found.")

lecture_df = pd.read_csv(LECTURE_TOPICS_CSV)

# Remove empty and clean up
lecture_df = lecture_df.dropna(subset=["Lecture Topic"])
lecture_df["Lecture Topic"] = lecture_df["Lecture Topic"].str.strip()
lecture_df["Subtopics"] = lecture_df["Subtopics"].fillna("").apply(lambda x: [s.strip() for s in x.split(";") if s.strip()])

# Create topic hierarchy with deduplication
hierarchy = {COURSE_NAME: {}}

for _, row in lecture_df.iterrows():
    topic = row["Lecture Topic"]
    subtopics = row["Subtopics"]

    if topic not in hierarchy[COURSE_NAME]:
        hierarchy[COURSE_NAME][topic] = set()

    hierarchy[COURSE_NAME][topic].update(subtopics)

# Convert sets to lists and remove duplicates
cleaned_hierarchy = {
    "course": COURSE_NAME,
    "topics": []
}

for topic, subtopic_set in hierarchy[COURSE_NAME].items():
    cleaned_hierarchy["topics"].append({
        "topic": topic,
        "subtopics": sorted(list(set(subtopic_set)))
    })

# Save to JSON
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(cleaned_hierarchy, f, indent=4, ensure_ascii=False)

print(f"✅ Course hierarchy JSON saved to {OUTPUT_JSON}")
