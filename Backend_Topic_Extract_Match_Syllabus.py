import fitz  # PyMuPDF
import os
import pandas as pd
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from numpy import dot
from numpy.linalg import norm

# --- Configuration ---
API_KEY = ""
SOURCE_FOLDER = "/workspaces/swayam/files"
OUTPUT_CSV = "/workspaces/swayam/result/New_topics_subtopics_parallel_output.csv"
SUBJECT_CONTEXT = ""
SYLLABUS_FOLDER = "/workspaces/swayam/syllabus"
OUTPUT_MATCHING_CSV = "/workspaces/swayam/result/topics_subtopics_matching_compare_results.csv"

client = OpenAI(api_key=API_KEY)

def build_prompt(text, subject=None):
    base = (
        "You are an academic assistant. Based on the following educational material, extract:\n"
        "1. A short lecture topic (4–8 words)\n"
        "2. 3–5 relevant subtopics.\n"
    )
    if subject:
        base += f"The subject area is {subject}.\n"
    base += f"\nCourse Content:\n\"\"\"\n{text[:2500]}\n\"\"\"\n\nOutput Format:\nTopic: <short topic>\nSubtopics:\n- <subtopic1>\n- <subtopic2>\n- ..."
    return base

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

def process_file(file_path):
    try:
        content = extract_text_from_pdf(file_path)
        prompt = build_prompt(content, SUBJECT_CONTEXT)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=200
        )
        output = response.choices[0].message.content.strip()
        topic = "-"
        subtopics_list = []
        for line in output.splitlines():
            if line.lower().startswith("topic:"):
                topic = line.split(":", 1)[-1].strip()
            elif line.strip().startswith("-"):
                subtopics_list.append(line.strip("- ").strip())
        return {
            "Source File": os.path.basename(file_path),
            "Lecture Topic": topic,
            "Subtopics": "; ".join(subtopics_list)
        }
    except Exception as e:
        return {
            "Source File": os.path.basename(file_path),
            "Lecture Topic": f"Error: {e}",
            "Subtopics": "-"
        }

def keyword_match_score(text1, text2):
    try:
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())
        if not set1 or not set2:
            return 0.0
        return round(len(set1 & set2) / len(set1 | set2) * 100, 2)
    except:
        return 0.0

def match_with_syllabus(row, syllabus_emb):
    best_sem_score = 0.0
    best_kw_score = 0.0
    best_match = "-"
    try:
        topic = row["Lecture Topic"]
        topic_emb = client.embeddings.create(model="text-embedding-ada-002", input=[topic]).data[0].embedding
        for s_text, s_emb in syllabus_emb:
            sem_score = dot(topic_emb, s_emb) / (norm(topic_emb) * norm(s_emb)) * 100
            kw_score = keyword_match_score(topic, s_text)
            if sem_score > best_sem_score:
                best_sem_score = sem_score
                best_match = s_text
            if kw_score > best_kw_score:
                best_kw_score = kw_score
    except:
        pass
    row["Best Match Syllabus Topic"] = best_match
    row["Semantic Match Score (%)"] = round(best_sem_score, 2)
    row["Keyword Match Score (%)"] = round(best_kw_score, 2)
    return row

if __name__ == "__main__":
    results = []

    if os.path.isfile(SOURCE_FOLDER) and SOURCE_FOLDER.endswith(".pdf"):
        results.append(process_file(SOURCE_FOLDER))
    elif os.path.isdir(SOURCE_FOLDER):
        pdf_files = [os.path.join(SOURCE_FOLDER, f) for f in os.listdir(SOURCE_FOLDER) if f.endswith(".pdf")]
        print(f"\nFound {len(pdf_files)} PDF files. Starting parallel processing...\n")
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_file = {executor.submit(process_file, file_path): file_path for file_path in pdf_files}
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    print(f"✅ Processed {file_path}")
                    results.append(result)
                except Exception as exc:
                    print(f"❌ {file_path} generated an exception: {exc}")
                    results.append({
                        "Source File": os.path.basename(file_path),
                        "Lecture Topic": f"Error: {exc}",
                        "Subtopics": "-"
                    })
    else:
        print("Invalid file or folder path provided.")

    if results:
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n✅ All Topics and Subtopics saved to {OUTPUT_CSV}")

        if os.path.isdir(SYLLABUS_FOLDER):
            syllabus_topics = []
            for file in os.listdir(SYLLABUS_FOLDER):
                if file.endswith(".pdf"):
                    content = extract_text_from_pdf(os.path.join(SYLLABUS_FOLDER, file))
                    lines = [line.strip() for line in content.splitlines() if line.strip() and len(line.strip()) > 4]
                    syllabus_topics.extend(lines)

            # TEMP FIX: limit lines for testing/performance
            syllabus_topics = syllabus_topics[:100]

            syllabus_emb = []
            for line in syllabus_topics:
                emb = client.embeddings.create(model="text-embedding-ada-002", input=[line]).data[0].embedding
                syllabus_emb.append((line, emb))

            print(f"\n🔍 Starting parallel syllabus matching for {len(results)} topics...\n")
            with ThreadPoolExecutor(max_workers=5) as executor:
                matched_rows = list(executor.map(lambda row: match_with_syllabus(row, syllabus_emb), results))

            df_matched = pd.DataFrame(matched_rows)
            df_matched.to_csv(OUTPUT_MATCHING_CSV, index=False)
            print(f"✅ Matching output saved to {OUTPUT_MATCHING_CSV}")
        else:
            print("⚠️ Syllabus folder not found or empty.")
    else:
        print("No PDFs found to process.")
