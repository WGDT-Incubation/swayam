# ai_edu_video_recommender/app.py

import streamlit as st
import requests
from youtube_transcript_api import YouTubeTranscriptApi
from sentence_transformers import SentenceTransformer, util
import faiss
import pickle

# Initialize
st.set_page_config(page_title="EduVideo Recommender", layout="centered")
st.title("🎓 AI-Powered Educational Video Finder")

# Load model & data
@st.cache_resource

def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Sample Syllabus Topics (In real version, scrape from DU or upload PDF)
topics = [
    "Balance of Payments",
    "Fiscal Policy and Budgeting",
    "National Income Accounting",
    "Inflation Causes and Control"
]

# Sample YouTube Video Links and Mock Transcripts (to simulate POC)
video_data = {
    "https://youtu.be/ABC123": "Balance of Payments refers to the record of all economic transactions...",
    "https://youtu.be/DEF456": "Fiscal policy involves government spending and taxation...",
    "https://youtu.be/GHI789": "National income refers to the total income earned by a country...",
    "https://youtu.be/JKL321": "Inflation is the rise in general price levels and is caused by demand..."
}

# Embed all transcripts
@st.cache_data

def embed_transcripts(video_data):
    texts = list(video_data.values())
    embeddings = model.encode(texts, convert_to_tensor=False)
    return embeddings

embeddings = embed_transcripts(video_data)

# Build FAISS Index
dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

video_links = list(video_data.keys())

# UI for Topic Selection
selected_topic = st.selectbox("Select a syllabus topic:", topics)

if st.button("🔍 Find Matching Videos"):
    topic_embedding = model.encode(selected_topic, convert_to_tensor=False)
    D, I = index.search([topic_embedding], k=3)

    st.subheader("Top Matching Videos:")
    for i in I[0]:
        st.write(f"▶️ [{video_links[i]}]({video_links[i]})")
        st.caption(video_data[video_links[i]][:100] + "...")
