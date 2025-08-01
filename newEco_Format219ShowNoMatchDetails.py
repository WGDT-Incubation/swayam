import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load local embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Using local model: all-MiniLM-L6-v2")

# ------------------- UTILITY FUNCTIONS -------------------
def embed(text: str) -> List[float]:
    return model.encode([text])[0]

def similarity_score(text1: str, text2: str) -> float:
    emb1 = embed(text1)
    emb2 = embed(text2)
    return float(cosine_similarity([emb1], [emb2])[0][0]) * 100

def load_json(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data: dict, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

# ------------------- FUNCTION 1: MATCH TOPICS -------------------
def match_topics(syllabus_json: dict, lecture_json: dict, course_name: str) -> List[Dict]:
    result = []
    for s_topic in syllabus_json["topics"]:
        s_text = f"Course: {course_name}, Topic: {s_topic['topic']}"
        best_match = {"similarity": 0}
        for l_topic in lecture_json["topics"]:
            l_text = f"Course: {course_name}, Topic: {l_topic['topic']}"
            score = similarity_score(s_text, l_text)
            if score > best_match.get("similarity", 0):
                best_match = {
                    "syllabus_topic": s_topic['topic'],
                    "matched_lecture_topic": l_topic['topic'],
                    "similarity": round(score, 2)
                }
        result.append(best_match)
    save_json(result, "/workspaces/swayam/result/Eco/2607_MU_matched_topics_local.json")
    return result

# ------------------- FUNCTION 2: MATCH SUBTOPICS -------------------
def match_subtopics(syllabus_json: dict, lecture_json: dict, course_name: str, with_context: bool = True) -> List[Dict]:
    results = []
    hierarchical_result = {"course": course_name, "topics": []}

    def match_pair(s_topic, s_sub):
        s_text = f"Course: {course_name}, Topic: {s_topic['topic']}, Subtopic: {s_sub}" if with_context else s_sub
        best = {"similarity": 0}
        for l_topic in lecture_json["topics"]:
            for l_sub in l_topic["subtopics"]:
                l_text = f"Course: {course_name}, Topic: {l_topic['topic']}, Subtopic: {l_sub}" if with_context else l_sub
                score = similarity_score(s_text, l_text)
                if score > best["similarity"]:
                    best = {
                        "syllabus_topic": s_topic['topic'],
                        "syllabus_subtopic": s_sub,
                        "matched_lecture_topic": l_topic['topic'],
                        "matched_lecture_subtopic": l_sub,
                        "similarity": round(score, 2)
                    }
        return s_topic['topic'], best

    with ThreadPoolExecutor() as executor:
        futures = []
        for s_topic in syllabus_json["topics"]:
            topic_node = {"topic": s_topic['topic'], "subtopics": []}

            if not s_topic.get("subtopics"):
                topic_node["subtopics"].append({
                    "syllabus_subtopic": "—",
                    "matched_lecture_topic": "—",
                    "matched_lecture_subtopic": "No matching — subtopics not provided",
                    "similarity": 0
                })
            else:
                for s_sub in s_topic["subtopics"]:
                    futures.append(executor.submit(match_pair, s_topic, s_sub))
            hierarchical_result["topics"].append(topic_node)

        for future in tqdm(futures):
            topic_title, match = future.result()
            for t in hierarchical_result["topics"]:
                if t["topic"] == topic_title:
                    t["subtopics"].append(match)

    save_json(hierarchical_result, "/workspaces/swayam/result/Eco/2607_MU_matched_subtopics_local.json")
    return hierarchical_result

# ------------------- FUNCTION 3: REPORTING -------------------
def generate_summary(match_results: List[Dict], key="similarity", label="Topic"):
    flat_results = []
    if isinstance(match_results, dict) and "topics" in match_results:
        for topic in match_results["topics"]:
            if "subtopics" in topic:
                for sub in topic["subtopics"]:
                    flat_results.append({
                        "topic": topic["topic"],
                        "subtopic": sub.get("syllabus_subtopic", ""),
                        "matched_lecture_topic": sub.get("matched_lecture_topic", ""),
                        "matched_lecture_subtopic": sub.get("matched_lecture_subtopic", ""),
                        "similarity": sub.get("similarity", 0)
                    })
    else:
        flat_results = match_results

    df = pd.DataFrame(flat_results)
    df.to_csv(f"summary_{label.lower()}.csv", index=False)

    matched_50 = df[df["similarity"] >= 50].shape[0]
    matched_75 = df[df["similarity"] >= 75].shape[0]
    matched_85 = df[df["similarity"] >= 85].shape[0]
    matched_95 = df[df["similarity"] >= 95].shape[0]
    total = df.shape[0]

    print(f"\nSummary for {label}s")
    print("Threshold\tMatches\tNo-match\t% Coverage")

    print(f"\u2265 95 %\t{matched_95}\t{total - matched_95}\t{round(matched_95 / total * 100, 2)}%")
    print(f"\u2265 85 %\t{matched_85}\t{total - matched_85}\t{round(matched_85 / total * 100, 2)}%")

    print(f"\u2265 75 %\t{matched_75}\t{total - matched_75}\t{round(matched_75 / total * 100, 2)}%")
    print(f"\u2265 50 %\t{matched_50}\t{total - matched_50}\t{round(matched_50 / total * 100, 2)}%")
    
    print(f"All\t{total}\t\u2014\t100%")
    
    df[df["similarity"] < 95].to_csv(f"mu_eco_no_match_{label.lower()}_95.csv", index=False)
    df[df["similarity"] < 85].to_csv(f"mu_eco_no_match_{label.lower()}_85.csv", index=False)
    df[df["similarity"] < 75].to_csv(f"mu_eco_no_match_{label.lower()}_75.csv", index=False)
    df[df["similarity"] < 50].to_csv(f"mu_eco_no_match_{label.lower()}_50.csv", index=False)

# ------------------- SAMPLE USAGE -------------------
if __name__ == '__main__':
    start = time.time()
    course = "Financial Accounting"
    syllabus_data = load_json("/workspaces/swayam/result/Eco/raw/MU_Economic_Growth_and_Development_syllabus0107.json")  #bharatividyapeeth
    lecture_data = load_json("/workspaces/swayam/result/Eco/2607_Eco_hierarchical_lecture_topics.json")

    topic_matches = match_topics(syllabus_data, lecture_data, course)
    generate_summary(topic_matches, key="similarity", label="Topic")

    subtopic_hierarchical = match_subtopics(syllabus_data, lecture_data, course, with_context=True)
    generate_summary(subtopic_hierarchical, key="similarity", label="Subtopic")

    print(f"\n✅ Finished in {round(time.time() - start, 2)}s")
