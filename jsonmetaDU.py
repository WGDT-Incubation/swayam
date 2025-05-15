import pandas as pd
import json
import os
from collections import defaultdict

# --- Configuration ---
CSV_FILE = "/workspaces/swayam/result/du1506_extracted_topics_subtopics_llm.csv"
OUTPUT_JSON = "/workspaces/swayam/result/du1506_nested_topics_subtopics.json"

# --- Read CSV and build JSON ---
def build_json_from_csv(csv_path):
    if not os.path.exists(csv_path):
        print(f"❌ CSV not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    topic_map = defaultdict(set)

    for _, row in df.iterrows():
        topic = str(row["Topic"]).strip()
        subtopic = str(row["Subtopic"]).strip()
        topic_map[topic].add(subtopic)

    json_structure = {
        "course": "Financial Accounting",
        "topics": []
    }

    for topic, subtopics in topic_map.items():
        json_structure["topics"].append({
            "topic": topic,
            "subtopics": sorted(list(subtopics))
        })

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(json_structure, f, indent=2, ensure_ascii=False)

    print(f"✅ Nested JSON saved to {OUTPUT_JSON}")

# --- Run ---
if __name__ == "__main__":
    build_json_from_csv(CSV_FILE)
