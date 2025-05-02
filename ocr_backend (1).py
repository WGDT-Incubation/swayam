import fitz  # PyMuPDF
import os
import pandas as pd
import tempfile
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import pytesseract

# --- Configuration ---
API_KEY = ""
SOURCE_FOLDER = "/workspaces/swayam/files"
OUTPUT_CSV = "/workspaces/swayam/result/gpt4_topics_subtopics_ocr_output.csv"
SUBJECT_CONTEXT = "Optional Subject Context like AI, Data Science"

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

def extract_text_with_ocr(doc):
    full_text = ""
    for page_num in range(len(doc)):
        pix = doc.load_page(page_num).get_pixmap(dpi=300)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as img_file:
            img_file.write(pix.tobytes("png"))
            img_path = img_file.name
        img = Image.open(img_path)
        text = pytesseract.image_to_string(img)
        full_text += text + "\n"
    return full_text

def extract_text_from_pdf(file_path):
    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        if len(text.strip()) < 100:
            print(f"🔍 Using OCR for: {file_path}")
            text = extract_text_with_ocr(doc)
        return text
    except Exception as e:
        print(f"OCR fallback failed for {file_path}: {e}")
        return ""

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

if __name__ == "__main__":
    results = []
    if os.path.isfile(SOURCE_FOLDER) and SOURCE_FOLDER.endswith(".pdf"):
        results.append(process_file(SOURCE_FOLDER))
    elif os.path.isdir(SOURCE_FOLDER):
        pdf_files = [os.path.join(SOURCE_FOLDER, f) for f in os.listdir(SOURCE_FOLDER) if f.endswith(".pdf")]
        print(f"📂 Found {len(pdf_files)} PDF files. Starting OCR + GPT-4 topic extraction...")
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_file = {executor.submit(process_file, file): file for file in pdf_files}
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    print(f"✅ Processed: {file_path}")
                    results.append(result)
                except Exception as e:
                    print(f"❌ Failed on: {file_path} → {e}")
                    results.append({
                        "Source File": os.path.basename(file_path),
                        "Lecture Topic": f"Error: {e}",
                        "Subtopics": "-"
                    })
    else:
        print("❗ Invalid file or folder path provided.")

    if results:
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"✅ Output saved to {OUTPUT_CSV}")
    else:
        print("No valid PDFs to process.")
