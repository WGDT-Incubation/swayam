import streamlit as st
import fitz  # PyMuPDF for PDF
import pytesseract
from PIL import Image
import io
import json

st.title("University Syllabus Topic-Subtopic JSON Generator")

# --- Basic Inputs ---
university = st.text_input("University Name")
course = st.text_input("Course Name")

# --- Upload Files for OCR ---
uploaded_files = st.file_uploader("Upload PDF/Image (Optional for OCR Extraction)", accept_multiple_files=True, type=["pdf", "png", "jpg", "jpeg"])
extracted_text = ""

if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file.name.lower().endswith(".pdf"):
            with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
                for page in doc:
                    extracted_text += page.get_text()
        else:
            image = Image.open(uploaded_file)
            text = pytesseract.image_to_string(image)
            extracted_text += text

    st.text_area("Extracted Text from Uploaded Files", extracted_text, height=200)

# --- Manual Topic/Subtopic Table Input ---
st.subheader("Enter Topics and Subtopics")
num_rows = st.number_input("Number of Topics", min_value=1, step=1, value=1)

syllabus_data = []

for i in range(num_rows):
    topic = st.text_input(f"Topic {i+1}", key=f"topic_{i}")
    subtopics = st.text_area(f"Subtopics for Topic {i+1} (separate by semicolon)", key=f"subs_{i}")
    if topic:
        subtopic_list = [s.strip() for s in subtopics.split(";") if s.strip()]
        syllabus_data.append({"topic": topic, "subtopics": subtopic_list})

# --- Generate JSON ---
if st.button("Generate JSON"):
    json_output = {
        "university": university,
        "course": course,
        "topics": syllabus_data
    }
    st.subheader("Generated JSON Metadata")
    st.json(json_output)

    st.download_button(
        label="Download JSON",
        data=json.dumps(json_output, indent=2),
        file_name=f"{course.replace(' ', '_')}_syllabus0107.json",
        mime="application/json"
    )
