
import fitz  # PyMuPDF
import re
import json

def extract_syllabus_topics(pdf_path, output_json_path, university="LU"):
    syllabus_data = {"course": "Financial Accounting", "topics": []}

    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()

    if university == "LU":
        # LU pattern: full unit content becomes topic, no subtopics
        units = re.findall(r"Unit\s+[IVX]+\s*:\s*(.*?)\n(?=Unit\s+[IVX]+\s*:|$)", full_text, re.DOTALL)
        for unit in units:
            sentences = re.split(r"[\n.]", unit)
            for s in sentences:
                s = s.strip()
                if s:
                    syllabus_data["topics"].append({"topic": s, "subtopics": []})

    elif university == "DU":
        # DU pattern: topic then roman or alphabet subtopics
        units = re.findall(r"Unit\s+\d+\s*:\s*(.*?)\n(?=Unit\s+\d+\s*:|$)", full_text, re.DOTALL)
        for unit in units:
            topic_match = re.match(r"(.*?)\n", unit)
            if topic_match:
                topic = topic_match.group(1).strip()
                subtopics = re.findall(r"\(?i?[a-zivx]+\)|-\s+)(.*?)(?=\(?[a-zivx]+\)|$)", unit)
                subtopics_cleaned = [s.strip() for _, s in subtopics if s.strip()]
                syllabus_data["topics"].append({"topic": topic, "subtopics": subtopics_cleaned})

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(syllabus_data, f, indent=2, ensure_ascii=False)

# Example usage:
# extract_syllabus_topics("LU_Syllabus_Financial_Accounting.pdf", "LU_syllabus_extracted.json", university="LU")
# extract_syllabus_topics("DU_Syllabus.pdf", "DU_syllabus_extracted.json", university="DU")
