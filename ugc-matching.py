"""
ugc_semantic_match.py
---------------------------------
• Loads:
    ├ UGC_syllabus_topics_subtopics.json
    └ UGC_lecture_topics_subtopics.json
• Computes semantic similarity with SBERT (all-MiniLM-L6-v2)
• Threshold: 50 %
• Writes:
    ├ UGC_final_matching_result.json
• Prints a stats table (topics / sub-topics matched & unmatched)
"""

import json
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer, util


# ---------- CONFIG ----------
SYLLABUS_JSON = "UGC_syllabus_topics_subtopics.json"
LECTURE_JSON  = "UGC_lecture_topics_subtopics.json"
OUTPUT_JSON   = "UGC_final_matching_result.json"
THRESHOLD     = 50.0      # percentage


# ---------- LOAD DATA ----------
syllabus = json.loads(Path(SYLLABUS_JSON).read_text(encoding="utf-8"))
lecture  = json.loads(Path(LECTURE_JSON).read_text(encoding="utf-8"))

# ---------- MODEL ----------
model = SentenceTransformer("all-MiniLM-L6-v2")

def similarity(a: str, b: str) -> float:
    e1 = model.encode(a, convert_to_tensor=True)
    e2 = model.encode(b, convert_to_tensor=True)
    return float(util.cos_sim(e1, e2)[0][0] * 100)


# ---------- MATCHING ----------
result = {"course": syllabus["course"], "topics": []}

topic_matched = 0
subtopic_total = 0
subtopic_matched = 0
matched_lecture_topics = set()
matched_lecture_subtopics = set()

for s in syllabus["topics"]:
    s_title = s["topic"]
    s_subs  = s.get("subtopics", [])
    entry = {"syllabus_topic": s_title, "matched_lecture_topics": []}

    for lec in lecture["topics"]:
        l_title = lec["topic"]
        score = similarity(s_title, l_title)

        if score >= THRESHOLD:
            topic_matched += 1
            matched_lecture_topics.add(l_title)

            detail = {
                "lecture_topic": l_title,
                "topic_semantic_match_score": round(score, 2),
                "matched_subtopics": []
            }

            if s_subs:                                     # syllabus has sub-topics
                for s_sub in s_subs:
                    subtopic_total += 1
                    best_sc, best_match = 0.0, None
                    for l_sub in lec["subtopics"]:
                        sc = similarity(s_sub, l_sub)
                        if sc > best_sc:
                            best_sc, best_match = sc, l_sub
                    if best_sc >= THRESHOLD:
                        subtopic_matched += 1
                        matched_lecture_subtopics.add(best_match)
                        detail["matched_subtopics"].append({
                            "lecture_subtopic": best_match,
                            "matched_with_syllabus_subtopic": s_sub,
                            "semantic_match_score": round(best_sc, 2)
                        })
                    else:
                        detail["matched_subtopics"].append({
                            "lecture_subtopic": "NO Suitable Match",
                            "matched_with_syllabus_subtopic": s_sub
                        })
            else:
                detail["matched_subtopics"] = "NO Suitable Match"

            entry["matched_lecture_topics"].append(detail)

    result["topics"].append(entry)


# ---------- SAVE JSON ----------
Path(OUTPUT_JSON).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\n✔ Saved: {OUTPUT_JSON}\n")

# ---------- STATS ----------
total_syl_topics = len(syllabus["topics"])
total_lec_topics = len(lecture["topics"])

total_lec_subs = sum(len(t["subtopics"]) for t in lecture["topics"])

stats = pd.DataFrame(
    [
        ["Syllabus Topics", total_syl_topics, topic_matched, total_syl_topics - topic_matched],
        ["Lecture Topics", total_lec_topics, len(matched_lecture_topics), total_lec_topics - len(matched_lecture_topics)],
        ["Syllabus Subtopics", subtopic_total, subtopic_matched, subtopic_total - subtopic_matched],
        ["Lecture Subtopics", total_lec_subs, len(matched_lecture_subtopics), total_lec_subs - len(matched_lecture_subtopics)]
    ],
    columns=["Category", "Total", "Matched", "Unmatched"]
)

print(stats.to_string(index=False))
