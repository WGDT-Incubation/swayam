import fitz  # PyMuPDF
import os
import pandas as pd
from openai import AzureOpenAI
import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# --- Azure OpenAI Configuration via Environment Variables ---
endpoint = os.getenv("ENDPOINT_URL", "https://guru-mitra-azure-instance.openai.azure.com/")
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o-mini")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY", "")

SOURCE_FOLDER = "/workspaces/swayam/files/fileset_Eco"
#OUTPUT_CSV = "/workspaces/swayam/files/fileset5/2506_BS_Alllecture_topics_output.csv"
OUTPUT_CSV = "/workspaces/swayam/files/fileset5/2607_Eco_Alllecture_topics_output.csv"

# Initialize Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=subscription_key,
    api_version="2025-01-01-preview",
)


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
        model=deployment,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
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


if __name__ == "__main__":
    results = []
    for f in os.listdir(SOURCE_FOLDER):
        if f.endswith(".pdf"):
            topic, subtopics = extract_lecture_topics(os.path.join(SOURCE_FOLDER, f))
            results.append({"Source File": f, "Lecture Topic": topic, "Subtopics": subtopics})

    df_topics = pd.DataFrame(results)
    df_topics.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Lecture topics saved to {OUTPUT_CSV}")
