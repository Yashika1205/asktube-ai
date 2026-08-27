import os
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import WebshareProxyConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


def get_video_id(url):
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    return None


def get_transcript(video_id):
    ytt = YouTubeTranscriptApi(
        proxy_config=WebshareProxyConfig(
            proxy_username=st.secrets.get("WEBSHARE_USERNAME", os.getenv("WEBSHARE_USERNAME")),
            proxy_password=st.secrets.get("WEBSHARE_PASSWORD", os.getenv("WEBSHARE_PASSWORD")),
        )
    )
    try:
        transcript_list = ytt.list(video_id)
        try:
            transcript = transcript_list.find_transcript(["en"])
        except:
            transcript = list(transcript_list)[0]
        fetched = transcript.fetch()
        chunks_with_time = [
            {"text": chunk.text, "start": chunk.start}
            for chunk in fetched
        ]
    except Exception as e:
        raise Exception(f"Could not fetch transcript: {e}")

    full_text = " ".join(c["text"] for c in chunks_with_time)
    return full_text, chunks_with_time


def build_vectorstore(video_id):
    full_text, chunks_with_time = get_transcript(video_id)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    docs = splitter.create_documents([full_text])
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore, full_text, chunks_with_time


def format_timestamp(seconds):
    mins = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{mins:02d}:{secs:02d}"


def generate_summary(full_text, llm):
    from langchain_core.messages import HumanMessage
    prompt = f"""Given this YouTube video transcript, provide:
1. A brief summary (3-4 sentences)
2. 5 key points from the video

Transcript:
{full_text[:4000]}

Format your response as:
SUMMARY:
<summary here>

KEY POINTS:
- point 1
- point 2
- point 3
- point 4
- point 5"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


def search_by_topic(topic, chunks_with_time):
    results = []
    topic_lower = topic.lower()
    for chunk in chunks_with_time:
        if topic_lower in chunk["text"].lower():
            results.append(chunk)
    return results[:5]
