import fitz  # PyMuPDF
import os
import pandas as pd
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from numpy import dot
from numpy.linalg import norm
import nltk
from nltk import pos_tag, word_tokenize
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# --- Configuration ---
API_KEY = "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"
LECTURE_TOPICS_CSV = "/workspaces/swayam/result/74New_topics_subtopics_parallel_output.csv"
SYLLABUS_FILE = "/workspaces/swayam/syllabus/DU-10072015_Annexure-64.pdf"
OUTPUT_MATCHING_CSV = "/workspaces/swayam/result/1305du_topic_subtopic_match.csv"

client = OpenAI(api_key=API_KEY)

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

def extract_syllabus_unit_topics(text):
    lines = text.splitlines()
    units = {}
    current_unit = ""
    capture = False
    for line in lines:
        if line.strip().lower().startswith("unit i"):
            current_unit = "Unit I"
            units[current_unit] = []
            capture = True
        elif line.strip().lower().startswith("unit ii"):
            current_unit = "Unit II"
            units[current_unit] = []
        elif line.strip().lower().startswith("unit iii"):
            current_unit = "Unit III"
            units[current_unit] = []
        elif line.strip().lower().startswith("unit iv"):
            current_unit = "Unit IV"
            units[current_unit] = []
        elif current_unit and line.strip() != "" and not line.strip().startswith("Paper"):
            if any(c.isalpha() for c in line):
                parts = [p.strip() for p in line.split(",") if len(p.strip()) > 2]
                units[current_unit].extend(parts)
    return units

def keyword_match_score(text1, text2):
    try:
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())
        if not set1 or not set2:
            return 0.0
        return round(len(set1 & set2) / len(set1 | set2) * 100, 2)
    except:
        return 0.0

def semantic_match_score(text1, text2):
    try:
        emb1 = client.embeddings.create(model="text-embedding-ada-002", input=[text1]).data[0].embedding
        emb2 = client.embeddings.create(model="text-embedding-ada-002", input=[text2]).data[0].embedding
        return round(dot(emb1, emb2) / (norm(emb1) * norm(emb2)) * 100, 2)
    except:
        return 0.0

if __name__ == "__main__":
    # Load Lecture Topics
    df_lectures = pd.read_csv(LECTURE_TOPICS_CSV)
    df_lectures = df_lectures.dropna(subset=['Lecture Topic'])

    # Load and extract Syllabus Topics from Units I–IV only
    syllabus_text = extract_text_from_pdf(SYLLABUS_FILE)
    syllabus_units = extract_syllabus_unit_topics(syllabus_text)

    # Match each topic with syllabus subtopics
    match_results = []
    for _, row in df_lectures.iterrows():
        lecture_topic = row['Lecture Topic']
        best_unit = "-"
        best_subtopic = "-"
        best_sem = 0.0
        best_kw = 0.0

        for unit, subtopics in syllabus_units.items():
            for st in subtopics:
                sem_score = semantic_match_score(lecture_topic, st)
                kw_score = keyword_match_score(lecture_topic, st)
                if sem_score > best_sem:
                    best_sem = sem_score
                    best_unit = unit
                    best_subtopic = st
                if kw_score > best_kw:
                    best_kw = kw_score

        match_results.append({
            "Lecture Topic": lecture_topic,
            "Best Matched Unit": best_unit,
            "Matched Syllabus Subtopic": best_subtopic,
            "Semantic Match Score": best_sem,
            "Keyword Match Score": best_kw
        })

    df_out = pd.DataFrame(match_results)
    df_out.to_csv(OUTPUT_MATCHING_CSV, index=False)
    print(f"✅ DU Syllabus Matching Completed. Output saved to {OUTPUT_MATCHING_CSV}")
