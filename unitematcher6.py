import fitz  # PyMuPDF
import os
import pandas as pd
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from numpy import dot
from numpy.linalg import norm
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# --- Configuration ---
API_KEY = ""
SOURCE_FOLDER = "/workspaces/swayam/files"
OUTPUT_CSV = "/workspaces/swayam/result/1105lecture_topics_output.csv"
SYLLABUS_FILE = "/workspaces/swayam/syllabus/LU Syllabus Financial Accounting.pdf"
OUTPUT_MATCHING_CSV = "/workspaces/swayam/result/1105topic_matching_results.csv"

client = OpenAI(api_key=API_KEY)


def build_prompt(text):
    return (
        "You are an academic assistant. From the following content, extract a short lecture topic (4–8 words) "
        "and 3–5 related subtopics.\n"
        "Output Format:\nTopic: <short topic>\nSubtopics:\n- <subtopic1>\n- <subtopic2>\n- ...\n"
        f"\n\nCourse Content:\n\"\"\"\n{text[:2500]}\n\"\"\""
    )


def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    return "\n".join(page.get_text() for page in doc)


def extract_lecture_topics(file_path):
    content = extract_text_from_pdf(file_path)
    prompt = build_prompt(content)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=300
    )
    lines = response.choices[0].message.content.strip().splitlines()
    topic, subtopics = "-", []
    for line in lines:
        if line.lower().startswith("topic"):
            topic = line.split(":", 1)[1].strip()
        elif line.strip().startswith("-"):
            subtopics.append(line.strip("- ").strip())
    return topic, "; ".join(subtopics)


def extract_unitwise_topics(pdf_path):
    content = extract_text_from_pdf(pdf_path)
    lines = content.splitlines()
    units = {}
    current_unit = ""
    relevant = False

    for line in lines:
        line = line.strip()
        if "Paper I : Financial Accounting" in line:
            relevant = True
        elif relevant and line.startswith("Paper") and "Financial Accounting" not in line:
            break
        elif relevant and line.startswith("Unit"):
            if current_unit:
                parts = current_unit.split(":", 1)
                if len(parts) == 2:
                    unit_name = parts[0].strip()
                    topics = [t.strip() for t in parts[1].split(",") if len(t.strip()) > 2]
                    units[unit_name] = topics
            current_unit = line
        elif relevant and current_unit:
            current_unit += " " + line

    if current_unit:
        parts = current_unit.split(":", 1)
        if len(parts) == 2:
            unit_name = parts[0].strip()
            topics = [t.strip() for t in parts[1].split(",") if len(t.strip()) > 2]
            units[unit_name] = topics

    return units


def semantic_similarity(text1, text2):
    try:
        emb1 = client.embeddings.create(model="text-embedding-ada-002", input=[text1]).data[0].embedding
        emb2 = client.embeddings.create(model="text-embedding-ada-002", input=[text2]).data[0].embedding
        return round(dot(emb1, emb2) / (norm(emb1) * norm(emb2)) * 100, 2)
    except:
        return 0.0


def keyword_similarity(text1, text2):
    set1, set2 = set(text1.lower().split()), set(text2.lower().split())
    return round(len(set1 & set2) / len(set1 | set2) * 100, 2) if set1 and set2 else 0.0


if __name__ == "__main__":
    results = []
    for f in os.listdir(SOURCE_FOLDER):
        if f.endswith(".pdf"):
            topic, subtopics = extract_lecture_topics(os.path.join(SOURCE_FOLDER, f))
            results.append({"Source File": f, "Lecture Topic": topic, "Subtopics": subtopics})

    df_topics = pd.DataFrame(results)
    df_topics.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Lecture topics saved to {OUTPUT_CSV}")

    syllabus_units = extract_unitwise_topics(SYLLABUS_FILE)

    matched = []
    for row in results:
        topic = row["Lecture Topic"]
        match_row = {"Lecture Topic": topic}

        for unit, items in syllabus_units.items():
            for i, item in enumerate(items):
                sem_score = semantic_similarity(topic, item)
                kw_score = keyword_similarity(topic, item)
                match_row[f"{unit} - Topic {i+1}"] = item
                match_row[f"{unit} - SemMatch {i+1} (%)"] = sem_score
                match_row[f"{unit} - KeyMatch {i+1} (%)"] = kw_score

        matched.append(match_row)

    pd.DataFrame(matched).to_csv(OUTPUT_MATCHING_CSV, index=False)
    print(f"✅ Expanded matching results saved to {OUTPUT_MATCHING_CSV}")
