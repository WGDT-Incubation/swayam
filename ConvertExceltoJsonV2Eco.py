import pandas as pd
import json
import unicodedata
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
import numpy as np

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text)
    return text.encode("ascii", "ignore").decode("ascii").strip().lower()

def excel_to_semantic_json(
    file_path: str,
    course_name: str,
    topic_col: str = "Lecture Topic",
    subtopic_col: str = "Subtopics",
    separator: str = ";",
    json_out_path: str = "/workspaces/swayam/result/Eco/2607_Eco_hierarchical_lecture_topics.json",
    threshold: float = 0.90  # cosine similarity threshold
):
    df = pd.read_excel(file_path) if file_path.endswith(".xlsx") else pd.read_csv(file_path)
    df = df.dropna(subset=[topic_col]).fillna("")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    topic_embeddings = {}
    topic_subtopic_map = defaultdict(set)
    topic_alias_map = {}

    duplicate_topics = 0
    duplicate_subtopics = 0

    for _, row in df.iterrows():
        topic = clean_text(row[topic_col])
        subtopics_raw = clean_text(row[subtopic_col])
        subtopics = [s.strip() for s in subtopics_raw.split(separator) if s.strip()]
        subtopics_set = set(subtopics)

        topic_emb = model.encode(topic, convert_to_tensor=True)

        matched_topic = None
        for existing_topic, existing_emb in topic_embeddings.items():
            sim = float(util.cos_sim(topic_emb, existing_emb))
            if sim >= threshold:
                matched_topic = existing_topic
                duplicate_topics += 1
                break

        if matched_topic:
            for sub in subtopics_set:
                if sub in topic_subtopic_map[matched_topic]:
                    duplicate_subtopics += 1
                topic_subtopic_map[matched_topic].add(sub)
        else:
            topic_embeddings[topic] = topic_emb
            topic_subtopic_map[topic] = subtopics_set

    final_json = {
        "course": course_name,
        "topics": []
    }

    for topic, subtopics in topic_subtopic_map.items():
        final_json["topics"].append({
            "topic": topic,
            "subtopics": sorted(list(subtopics))
        })

    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=2)

    print(f"✅ JSON saved at: {json_out_path}")
    print("\n📊 Deduplication Stats:")
    print(f"• Total unique topics: {len(topic_subtopic_map)}")
    print(f"• Duplicate topic entries: {duplicate_topics}")
    print(f"• Total unique subtopics: {sum(len(s) for s in topic_subtopic_map.values())}")
    print(f"• Duplicate subtopic entries: {duplicate_subtopics}")

    return final_json

# Example usage
if __name__ == "__main__":
    file_path = "files/fileset5/2607_Eco_Alllecture_topics_output.csv"  # Update as needed
    course_name = "BUSINESS STATISTICS"
    excel_to_semantic_json(file_path, course_name)
