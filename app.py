import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from utils import (
    get_video_id, build_vectorstore,
    format_timestamp, generate_summary, search_by_topic
)

load_dotenv()

st.set_page_config(page_title="AskTube", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

* { font-family: 'Inter', sans-serif; }

/* Animated background */
.stApp {
    background: linear-gradient(135deg, #0a0a0f 0%, #0d0d1a 50%, #0a0f1a 100%);
    color: #e0e0e0;
}

/* Animated gradient orbs */
.stApp::before {
    content: '';
    position: fixed;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse at 20% 50%, rgba(120, 40, 200, 0.08) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 20%, rgba(60, 100, 255, 0.08) 0%, transparent 50%),
                radial-gradient(ellipse at 50% 80%, rgba(200, 40, 120, 0.05) 0%, transparent 50%);
    animation: orbFloat 15s ease infinite alternate;
    pointer-events: none;
    z-index: 0;
}

@keyframes orbFloat {
    0% { transform: translate(0, 0) rotate(0deg); }
    100% { transform: translate(2%, 2%) rotate(5deg); }
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(15, 10, 30, 0.95) !important;
    border-right: 1px solid rgba(139, 92, 246, 0.2) !important;
    backdrop-filter: blur(20px);
}

section[data-testid="stSidebar"] > div {
    padding-top: 2rem;
}

/* Main title */
.main-title {
    text-align: center;
    padding: 2rem 0 0.5rem 0;
}

.main-title h1 {
    font-size: 3.5rem;
    font-weight: 900;
    background: linear-gradient(135deg, #a855f7 0%, #ec4899 50%, #6366f1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: titlePulse 3s ease infinite alternate;
    letter-spacing: -2px;
    margin: 0;
}

@keyframes titlePulse {
    0% { filter: brightness(1); }
    100% { filter: brightness(1.2); }
}

.main-title .subtitle {
    color: #6b7280;
    font-size: 1rem;
    font-weight: 500;
    letter-spacing: 0.5px;
    margin-top: 0.3rem;
}

/* Glowing divider */
.glow-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #a855f7, #ec4899, #6366f1, transparent);
    margin: 1.5rem 0;
    animation: glowPulse 2s ease infinite alternate;
}

@keyframes glowPulse {
    0% { opacity: 0.5; }
    100% { opacity: 1; box-shadow: 0 0 10px rgba(168, 85, 247, 0.5); }
}

/* Sidebar heading */
.sidebar-heading {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #a855f7;
    margin-bottom: 0.8rem;
}

/* Input fields */
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(139, 92, 246, 0.3) !important;
    border-radius: 14px !important;
    color: white !important;
    padding: 0.7rem 1rem !important;
    transition: all 0.3s ease !important;
    font-size: 0.9rem !important;
}

.stTextInput > div > div > input:focus {
    border-color: rgba(168, 85, 247, 0.8) !important;
    box-shadow: 0 0 20px rgba(168, 85, 247, 0.2) !important;
    background: rgba(255,255,255,0.07) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #a855f7 0%, #ec4899 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.6rem 1.5rem !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
    letter-spacing: 0.5px;
    position: relative;
    overflow: hidden;
}

.stButton > button::after {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: linear-gradient(135deg, transparent 30%, rgba(255,255,255,0.1) 50%, transparent 70%);
    transform: translateX(-100%) rotate(45deg);
    transition: transform 0.5s ease;
}

.stButton > button:hover::after {
    transform: translateX(100%) rotate(45deg);
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(168, 85, 247, 0.4) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03);
    border-radius: 16px;
    padding: 6px;
    gap: 4px;
    border: 1px solid rgba(255,255,255,0.06);
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 12px;
    color: #6b7280;
    font-weight: 600;
    font-size: 0.85rem;
    padding: 0.5rem 1.2rem;
    transition: all 0.3s ease;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #a855f7, #ec4899) !important;
    color: white !important;
    box-shadow: 0 4px 15px rgba(168, 85, 247, 0.3);
}

/* Chat messages */
.stChatMessage {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 20px !important;
    padding: 1.2rem !important;
    margin: 0.6rem 0 !important;
    animation: fadeInUp 0.4s ease;
    backdrop-filter: blur(10px);
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(15px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Feature cards */
.feature-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(139, 92, 246, 0.2);
    border-radius: 16px;
    padding: 1.2rem;
    margin: 0.5rem 0;
    transition: all 0.3s ease;
    animation: fadeInUp 0.5s ease;
}

.feature-card:hover {
    border-color: rgba(168, 85, 247, 0.5);
    background: rgba(168, 85, 247, 0.05);
    transform: translateX(4px);
}

/* Timestamp badge */
.timestamp-badge {
    background: linear-gradient(135deg, #a855f7, #ec4899);
    color: white;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    margin-right: 10px;
    letter-spacing: 0.5px;
}

/* Source chunk */
.source-chunk {
    background: rgba(99, 102, 241, 0.08);
    border-left: 3px solid #a855f7;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.82rem;
    color: #9ca3af;
    line-height: 1.6;
}

/* Video container */
.video-container {
    display: flex;
    justify-content: center;
    margin: 1.5rem 0;
}

.video-container iframe {
    border-radius: 20px;
    box-shadow: 0 20px 60px rgba(168, 85, 247, 0.2);
    border: 1px solid rgba(168, 85, 247, 0.2);
}

/* Status badges */
.stSuccess {
    background: rgba(34, 197, 94, 0.1) !important;
    border: 1px solid rgba(34, 197, 94, 0.2) !important;
    border-radius: 12px !important;
}

.stError {
    background: rgba(239, 68, 68, 0.1) !important;
    border: 1px solid rgba(239, 68, 68, 0.2) !important;
    border-radius: 12px !important;
}

/* Loading spinner */
.stSpinner > div {
    border-top-color: #a855f7 !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(#a855f7, #ec4899);
    border-radius: 4px;
}

/* Chat input */
.stChatInput > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(139, 92, 246, 0.3) !important;
    border-radius: 16px !important;
}

.stChatInput > div:focus-within {
    border-color: rgba(168, 85, 247, 0.8) !important;
    box-shadow: 0 0 20px rgba(168, 85, 247, 0.15) !important;
}

/* Powered by badge */
.powered-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.75rem;
    color: #6b7280;
    margin: 0 4px;
}

/* Landing page hero */
.hero-section {
    text-align: center;
    padding: 4rem 2rem;
    animation: fadeInUp 0.6s ease;
}

.hero-emoji {
    font-size: 5rem;
    animation: bounce 2s ease infinite;
    display: block;
    margin-bottom: 1rem;
}

@keyframes bounce {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-10px); }
}

.hero-title {
    font-size: 2rem;
    font-weight: 800;
    color: #e0e0e0;
    margin-bottom: 0.5rem;
}

.hero-subtitle {
    color: #6b7280;
    font-size: 1rem;
    margin-bottom: 3rem;
}

/* Feature grid cards */
.grid-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(139, 92, 246, 0.15);
    border-radius: 20px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
    animation: fadeInUp 0.5s ease;
}

.grid-card:hover {
    border-color: rgba(168, 85, 247, 0.4);
    background: rgba(168, 85, 247, 0.05);
    transform: translateY(-4px);
    box-shadow: 0 10px 30px rgba(168, 85, 247, 0.15);
}

.grid-card-emoji {
    font-size: 2.5rem;
    margin-bottom: 0.8rem;
    display: block;
}

.grid-card-title {
    font-weight: 700;
    font-size: 1rem;
    color: #e0e0e0;
    margin-bottom: 0.4rem;
}

.grid-card-desc {
    font-size: 0.82rem;
    color: #6b7280;
    line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)


def get_llm():
    from langchain_groq import ChatGroq
    groq_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
    if not groq_key:
        raise ValueError("GROQ_API_KEY not found. Set it in Streamlit Secrets (cloud) or .env (local).")
    return ChatGroq(
        model="openai/gpt-oss-20b",
        api_key=groq_key
    )


def get_answer(query, vectorstore, llm):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6}
    )

    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant that answers questions about a YouTube video based on its transcript.
Use the context below to answer the question thoroughly and accurately.
If the answer is not in the context, say "This topic is not covered in the video."
Always give detailed, well-structured answers.

Context:
{context}

Question: {question}

Answer:
""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    docs = retriever.invoke(query)
    answer = chain.invoke(query)
    sources = [doc.page_content[:250] + "..." for doc in docs]
    return answer, sources


# ── Header ──
st.markdown("""
<div class='main-title'>
    <h1>⚡ AskTube</h1>
    <p class='subtitle'>Drop a YouTube link. Ask anything. Get answers instantly.</p>
</div>
<div class='glow-divider'></div>
""", unsafe_allow_html=True)


# ── Sidebar ──
with st.sidebar:
    st.markdown("<p class='sidebar-heading'>🎯 Load Video</p>", unsafe_allow_html=True)
    url = st.text_input("", placeholder="https://youtube.com/watch?v=...")
    load_btn = st.button("⚡ Analyze Video")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<p class='sidebar-heading'>🔍 Topic Search</p>", unsafe_allow_html=True)
    topic = st.text_input(" ", placeholder="e.g. machine learning")
    topic_btn = st.button("🔎 Search Topic")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
        <span class='powered-badge'>🦙 Llama 3.1</span>
        <span class='powered-badge'>🔍 RAG</span>
        <br><br>
        <span class='powered-badge'>⚡ FAISS</span>
        <span class='powered-badge'>🤗 HuggingFace</span>
    </div>
    """, unsafe_allow_html=True)


# ── Load Video ──
if load_btn and url:
    video_id = get_video_id(url)
    if not video_id:
        st.error("❌ Invalid YouTube URL.")
    else:
        try:
            with st.spinner("🔍 Fetching transcript..."):
                vectorstore, full_text, chunks_with_time = build_vectorstore(video_id)
                st.session_state.vectorstore = vectorstore
                st.session_state.full_text = full_text
                st.session_state.chunks_with_time = chunks_with_time
                st.session_state.messages = []
                st.session_state.video_id = video_id
            with st.spinner("🧠 Generating summary..."):
                llm = get_llm()
                summary = generate_summary(full_text, llm)
                st.session_state.summary = summary
            st.sidebar.success("✅ Ready to chat!")
        except Exception as e:
            st.error(f"❌ {e}")


# ── Main Content ──
if "vectorstore" in st.session_state:
    vid_id = st.session_state.video_id

    st.markdown(
        f'<div class="video-container">'
        f'<iframe width="680" height="382" src="https://www.youtube.com/embed/{vid_id}" '
        f'frameborder="0" allowfullscreen></iframe></div>',
        unsafe_allow_html=True
    )

    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📋 Summary", "⏱️ Timestamps"])

    with tab1:
        for msg in st.session_state.get("messages", []):
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if msg.get("sources"):
                    with st.expander("📄 Sources"):
                        for s in msg["sources"]:
                            st.markdown(f"<div class='source-chunk'>{s}</div>", unsafe_allow_html=True)

        query = st.chat_input("Ask anything about this video...")
        if query:
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.write(query)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    llm = get_llm()
                    answer, sources = get_answer(query, st.session_state.vectorstore, llm)
                    st.write(answer)
                    with st.expander("📄 Sources"):
                        for s in sources:
                            st.markdown(f"<div class='source-chunk'>{s}</div>", unsafe_allow_html=True)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

    with tab2:
        if "summary" in st.session_state:
            parts = st.session_state.summary.split("KEY POINTS:")
            if len(parts) == 2:
                summary_part = parts[0].replace("SUMMARY:", "").strip()
                keypoints_part = parts[1].strip()
                st.markdown("### 📝 Summary")
                st.markdown(f"<div class='feature-card'>{summary_part}</div>", unsafe_allow_html=True)
                st.markdown("### 🔑 Key Points")
                for line in keypoints_part.split("\n"):
                    if line.strip().startswith("-"):
                        st.markdown(
                            f"<div class='feature-card'>✅ {line.strip()[1:].strip()}</div>",
                            unsafe_allow_html=True
                        )
            else:
                st.write(st.session_state.summary)

    with tab3:
        st.markdown("#### Find exact moments in the video")
        if topic_btn and topic:
            results = search_by_topic(topic, st.session_state.chunks_with_time)
            if results:
                st.success(f"Found **{len(results)}** mentions of **'{topic}'**")
                for r in results:
                    ts = format_timestamp(r["start"])
                    yt_link = f"https://www.youtube.com/watch?v={vid_id}&t={int(r['start'])}s"
                    st.markdown(
                        f"<div class='feature-card'>"
                        f"<span class='timestamp-badge'>⏱ {ts}</span>"
                        f"{r['text']} "
                        f"<a href='{yt_link}' target='_blank' style='color:#a855f7; font-weight:600;'>▶ Jump</a>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
            else:
                st.warning(f"No mentions of **'{topic}'** found.")
        else:
            st.markdown("<p style='color:#6b7280;'>Search a topic from the sidebar to find timestamps!</p>", unsafe_allow_html=True)

else:
    # Landing page
    st.markdown("""
    <div class='hero-section'>
        <span class='hero-emoji'>⚡</span>
        <p class='hero-title'>Your AI YouTube Companion</p>
        <p class='hero-subtitle'>Paste any YouTube URL and start chatting with the video instantly</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class='grid-card'>
            <span class='grid-card-emoji'>💬</span>
            <p class='grid-card-title'>Smart Chat</p>
            <p class='grid-card-desc'>Ask anything about the video and get accurate, detailed answers</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='grid-card'>
            <span class='grid-card-emoji'>📋</span>
            <p class='grid-card-title'>Auto Summary</p>
            <p class='grid-card-desc'>Get instant summary and key points without watching the full video</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='grid-card'>
            <span class='grid-card-emoji'>⏱️</span>
            <p class='grid-card-title'>Timestamp Search</p>
            <p class='grid-card-desc'>Find exactly when any topic is discussed with clickable timestamps</p>
        </div>""", unsafe_allow_html=True)
