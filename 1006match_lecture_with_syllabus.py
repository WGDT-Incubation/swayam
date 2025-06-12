
import json
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_similarity(a, b):
    emb1 = model.encode(a, convert_to_tensor=True)
    emb2 = model.encode(b, convert_to_tensor=True)
    return util.pytorch_cos_sim(emb1, emb2).item() * 100

def match_lecture_with_syllabus(syllabus_path, lecture_path, output_path):
    with open(syllabus_path, 'r', encoding='utf-8') as f:
        syllabus = json.load(f)

    with open(lecture_path, 'r', encoding='utf-8') as f:
        lectures = json.load(f)

    result = {"course": syllabus["course"], "topics": []}

    for s_topic in syllabus["topics"]:
        match_entry = {"syllabus_topic": s_topic["topic"], "matched_lecture_topics": []}
        for lec_topic in lectures["topics"]:
            score = compute_similarity(s_topic["topic"], lec_topic["topic"])
            if score >= 50:
                match_obj = {
                    "lecture_topic": lec_topic["topic"],
                    "topic_semantic_match_score": round(score, 2),
                    "matched_subtopics": []
                }
                if s_topic["subtopics"]:
                    for s_sub in s_topic["subtopics"]:
                        sub_match_found = False
                        for lec_sub in lec_topic["subtopics"]:
                            sub_score = compute_similarity(s_sub, lec_sub)
                            if sub_score >= 50:
                                match_obj["matched_subtopics"].append({
                                    "lecture_subtopic": lec_sub,
                                    "matched_with_syllabus_subtopic": s_sub,
                                    "semantic_match_score": round(sub_score, 2)
                                })
                                sub_match_found = True
                        if not sub_match_found:
                            match_obj["matched_subtopics"].append({
                                "lecture_subtopic": "NO Suitable Match",
                                "matched_with_syllabus_subtopic": s_sub
                            })
                else:
                    match_obj["matched_subtopics"] = "NO Suitable Match"

                match_entry["matched_lecture_topics"].append(match_obj)
        result["topics"].append(match_entry)

    with open(output_path, 'w', encoding='utf-8') as out:
        json.dump(result, out, indent=2, ensure_ascii=False)

# Example usage:
# match_lecture_with_syllabus('DU_syllabus_extracted.json', 'DU_lecture_topics.json', 'DU_final_matched.json')
