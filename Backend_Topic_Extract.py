import fitz  # PyMuPDF
import os
import pandas as pd
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Configuration ---

API_KEY = ""
SOURCE_FOLDER = "/workspaces/swayam/files/book" #files for transcripts and book path for ebook
OUTPUT_CSV = "/workspaces/swayam/result/gpt4_topics_subtopics_parallel_output.csv"
SUBJECT_CONTEXT = ""  # Optional subject context - Economics

# --- Initialize OpenAI client ---
client = OpenAI(api_key=API_KEY)

# --- Build prompt ---
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

# --- Extract full text from PDF ---
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

# --- Process a single file ---
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

        # Parse topic and subtopics
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

# --- Main Execution ---
if __name__ == "__main__":
    results = []
    
    if os.path.isfile(SOURCE_FOLDER) and SOURCE_FOLDER.endswith(".pdf"):
        results.append(process_file(SOURCE_FOLDER))
    elif os.path.isdir(SOURCE_FOLDER):
        pdf_files = [os.path.join(SOURCE_FOLDER, f) for f in os.listdir(SOURCE_FOLDER) if f.endswith(".pdf")]

        print(f"\nFound {len(pdf_files)} PDF files. Starting parallel processing...\n")

        with ThreadPoolExecutor(max_workers=5) as executor:  # Tune max_workers as per CPU
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

    # Save output
    if results:
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n✅ All Topics and Subtopics saved to {OUTPUT_CSV}")
    else:
        print("No PDFs found to process.")
