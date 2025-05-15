import fitz  # PyMuPDF
import os
import pandas as pd
from openai import OpenAI

# --- Configuration ---
SYLLABUS_FILE = "/workspaces/swayam/syllabus/DU-10072015_Annexure-64.pdf"
OUTPUT_CSV = "/workspaces/swayam/result/du1506_extracted_topics_subtopics_llm.csv"
API_KEY = ""


# --- Initialize OpenAI Client ---
client = OpenAI(api_key=API_KEY)

# --- Extract Text from PDF ---
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text() + "\n"
    return full_text

# --- Extract Only the Financial Accounting Section ---
def extract_financial_accounting_block(text):
    lines = text.splitlines()
    start, end = -1, -1
    for i, line in enumerate(lines):
        if "Paper 1.2" in line and "Financial Accounting" in line:
            start = i
        if start != -1 and "Paper 1.3" in line:
            end = i
            break
    if start != -1 and end != -1:
        return "\n".join(lines[start:end])
    return ""

# --- Prompt Builder ---
def build_prompt(extracted_block):
    return (
        "You are an academic assistant. Below is a section of a Financial Accounting syllabus.\n"
        "Extract a clean table with:\n"
        "1. Unit (if mentioned)\n"
        "2. Topic (like 'Theoretical Framework', 'Accounting Process', etc.)\n"
        "3. Subtopics: derived from i., ii., iii. or paragraph under each topic\n\n"
        "Ignore lines like 'Lectures', 'Duration', and 'Marks'.\n"
        "Output format:\n"
        "Unit,Topic,Subtopic\n\n"
        "Example:\n"
        "Unit I,Theoretical Framework,Accounting as an information system, the users of financial accounting information...\n"
        "Unit I,Theoretical Framework,The nature of financial accounting principles – Basic concepts and conventions...\n"
        "Unit I,Theoretical Framework,Financial accounting standards: Concept, benefits, procedure for issuing standards...\n"
        "Unit I,Accounting Process,From recording of a business transaction to preparation of trial balance...\n\n"
        "Now extract the following:\n"
        f"\"\"\"\n{extracted_block}\n\"\"\""
    )


# --- GPT Call ---
def get_structured_topics(section):
    prompt = build_prompt(section)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1500
    )
    return response.choices[0].message.content.strip()

# --- CSV Writer ---
def save_to_csv(parsed_text):
    lines = parsed_text.splitlines()
    rows = []
    for line in lines:
        if line.lower().startswith("unit") and line.count(",") >= 2:
            unit, topic, subtopic = [x.strip() for x in line.split(",", 2)]
            rows.append({"Unit": unit, "Topic": topic, "Subtopic": subtopic})
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Final output saved to {OUTPUT_CSV}")

# --- Main ---
if __name__ == "__main__":
    full_text = extract_text_from_pdf(SYLLABUS_FILE)
    target_section = extract_financial_accounting_block(full_text)
    if not target_section:
        print("❌ Financial Accounting section not found!")
    else:
        print("🔍 Extracting with GPT...")
        parsed = get_structured_topics(target_section)
        print(parsed)
        save_to_csv(parsed)
