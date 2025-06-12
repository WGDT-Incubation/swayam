
import os
import openai
import json

openai.api_key = os.getenv("OPENAI_API_KEY")

def extract_topics_from_transcripts(transcript_folder, output_json_path):
    lectures = {}
    for filename in os.listdir(transcript_folder):
        if filename.endswith(".txt"):
            with open(os.path.join(transcript_folder, filename), 'r', encoding='utf-8') as file:
                content = file.read()
                prompt = f"""
You are an expert in analyzing academic transcripts. Extract the main topics and subtopics from the following Financial Accounting lecture transcript. Structure the result in JSON format where 'course' is 'Financial Accounting', 'topics' is a list of dictionaries, each with a 'topic' and a list of 'subtopics'.

Transcript:
\"\"\"{content}\"\"\"

Respond only with the JSON.
"""
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
                )
                data = response['choices'][0]['message']['content']
                try:
                    lectures[filename] = json.loads(data)
                except json.JSONDecodeError:
                    print(f"JSON decode error for file: {filename}")

    with open(output_json_path, 'w', encoding='utf-8') as outfile:
        json.dump(lectures, outfile, indent=2, ensure_ascii=False)

# Example usage
# extract_topics_from_transcripts('./transcripts', 'lecture_topics_output.json')
