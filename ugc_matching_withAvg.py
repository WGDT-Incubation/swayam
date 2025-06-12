
"""
ugc_semantic_match.py  (updated)
---------------------------------
• Loads syllabus & lecture JSON
• Calculates semantic similarity via SBERT
• Keeps THRESHOLD for deciding “matched_lecture_topics”
• Regardless of threshold, computes overall averages:
      average_topic_semantic_score
      average_subtopic_semantic_score
  (uses best score for every syllabus topic / sub‑topic)
• Writes UGC_final_matching_result.json with averages appended.
"""

import json
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

# ---------- CONFIG ----------
SYLLABUS_JSON = "UGC_syllabus_topics_subtopics.json"
LECTURE_JSON  = "UGC_lecture_topics_subtopics.json"
OUTPUT_JSON   = "UGC_final_matching_result.json"
THRESHOLD     = 50.0   # %  – used only to populate matched_lecture_topics list

# ---------- LOAD ----------
syllabus = json.loads(Path(SYLLABUS_JSON).read_text(encoding="utf-8"))
lecture  = json.loads(Path(LECTURE_JSON).read_text(encoding="utf-8"))

# ---------- MODEL ----------
model = SentenceTransformer("all-MiniLM-L6-v2")
def sim(a:str,b:str)->float:
    e1 = model.encode(a, convert_to_tensor=True)
    e2 = model.encode(b, convert_to_tensor=True)
    return float(util.cos_sim(e1,e2)[0][0]*100)

# ---------- MATCHING ----------
result = {"course": syllabus["course"], "topics": []}

topic_matched = 0
subtopic_total = 0
subtopic_matched = 0
matched_lecture_topics = set()
matched_lecture_subtopics = set()

topic_scores_for_avg = []
subtopic_scores_for_avg = []

for s in syllabus["topics"]:
    s_title = s["topic"]
    s_subs  = s.get("subtopics", [])
    entry = {"syllabus_topic": s_title, "matched_lecture_topics": []}

    # best score for averaging (topic level)
    best_topic_score = 0
    best_topic_lec   = None

    for lec in lecture["topics"]:
        l_title = lec["topic"]
        score = sim(s_title, l_title)

        # track best for averaging
        if score > best_topic_score:
            best_topic_score = score
            best_topic_lec   = l_title

        # include in output if above threshold
        if score >= THRESHOLD:
            topic_matched += 1
            matched_lecture_topics.add(l_title)

            detail = {
                "lecture_topic": l_title,
                "topic_semantic_match_score": round(score,2),
                "matched_subtopics": []
            }

            if s_subs:
                for s_sub in s_subs:
                    subtopic_total += 1
                    best_sc, best_match = 0.0, None
                    for l_sub in lec["subtopics"]:
                        sc = sim(s_sub, l_sub)
                        if sc > best_sc:
                            best_sc, best_match = sc, l_sub

                    # collect for average (best per syllabus subtopic)
                    if best_sc>0:
                        subtopic_scores_for_avg.append(best_sc)

                    if best_sc >= THRESHOLD:
                        subtopic_matched += 1
                        matched_lecture_subtopics.add(best_match)
                        detail["matched_subtopics"].append({
                            "lecture_subtopic": best_match,
                            "matched_with_syllabus_subtopic": s_sub,
                            "semantic_match_score": round(best_sc,2)
                        })
                    else:
                        detail["matched_subtopics"].append({
                            "lecture_subtopic": "NO Suitable Match",
                            "matched_with_syllabus_subtopic": s_sub
                        })
            else:
                detail["matched_subtopics"] = "NO Suitable Match"

            entry["matched_lecture_topics"].append(detail)

    # append topic best score for averaging
    if best_topic_lec is not None:
        topic_scores_for_avg.append(best_topic_score)

    result["topics"].append(entry)

# ---------- AVERAGES ----------
average_topic_score = round(sum(topic_scores_for_avg)/len(topic_scores_for_avg),2) if topic_scores_for_avg else 0
average_subtopic_score = round(sum(subtopic_scores_for_avg)/len(subtopic_scores_for_avg),2) if subtopic_scores_for_avg else 0
result["average_topic_semantic_score"] = average_topic_score
result["average_subtopic_semantic_score"] = average_subtopic_score

# ---------- SAVE ----------
Path(OUTPUT_JSON).write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
print(f"\n✔ Saved: {OUTPUT_JSON}\nAverage Topic Score: {average_topic_score}%\nAverage Subtopic Score: {average_subtopic_score}%")

# ---------- STATS ----------
total_syl_topics = len(syllabus["topics"])
total_lec_topics = len(lecture["topics"])
total_lec_subs   = sum(len(t["subtopics"]) for t in lecture["topics"])

stats = pd.DataFrame(
    [
        ["Syllabus Topics", total_syl_topics, topic_matched, total_syl_topics - topic_matched],
        ["Lecture Topics", total_lec_topics, len(matched_lecture_topics), total_lec_topics-len(matched_lecture_topics)],
        ["Syllabus Subtopics", subtopic_total, subtopic_matched, subtopic_total-subtopic_matched],
        ["Lecture Subtopics", total_lec_subs, len(matched_lecture_subtopics), total_lec_subs-len(matched_lecture_subtopics)],
        ["Average % (all)", "-", f"{average_topic_score}%", f"{average_subtopic_score}%"]
    ],
    columns=["Category","Total","Matched","Unmatched"]
)
print("\n"+stats.to_string(index=False))
