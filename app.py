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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif; }

.stApp {
    background: #FFF8F0;
    color: #4A3B32;
}

section[data-testid="stSidebar"] {
    display: none;
}

.main-title {
    text-align: center;
    padding: 1.5rem 0 0.5rem 0;
}

.main-title h1 {
    font-size: 2.2rem;
    font-weight: 500;
    color: #4A3B32;
    margin: 0;
}

.main-title .subtitle {
    color: #8B7355;
    font-size: 0.95rem;
    margin-top: 0.3rem;
}

.stTextInput > div > div > input {
    background: #FFFFFF !important;
    border: 1px solid #F0DCC8 !important;
    border-radius: 10px !important;
    color: #4A3B32 !important;
    padding: 0.7rem 1rem !important;
    font-size: 0.9rem !important;
}

.stTextInput > div > div > input:focus {
    border-color: #E76F51 !important;
    box-shadow: 0 0 0 2px rgba(231,111,81,0.15) !important;
}

.stButton > button {
    background: linear-gradient(135deg, #E76F51, #F4A261) !important;
    color: #FFF8F0 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.5rem !important;
    font-weight: 500 !important;
    width: 100% !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(231,111,81,0.3) !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: #FFEDE0;
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 9px;
    color: #8B7355;
    font-weight: 500;
    font-size: 0.85rem;
    padding: 0.5rem 1.2rem;
}

.stTabs [aria-selected="true"] {
    background: #FFFFFF !important;
    color: #C1502B !important;
}

.feature-card {
    background: #FFFFFF;
    border: 1px solid #F5DAD1;
    border-radius: 14px;
    padding: 1.1rem;
    margin: 0.5rem 0;
}

.timestamp-badge {
    background: linear-gradient(135deg, #E76F51, #F4A261);
    color: #FFF8F0;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 10px;
}

.source-chunk {
    background: #FBE4DC;
    border-left: 3px solid #E76F51;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.82rem;
    color: #8B6656;
}

.video-container {
    display: flex;
    justify-content: center;
    margin: 1.2rem 0;
}

.video-container iframe {
    border-radius: 16px;
    border: 1px solid #F5DAD1;
}

.stSuccess {
    background: #EAF3DE !important;
    border-radius: 10px !important;
}

.stError {
    background: #FCEBEB !important;
    border-radius: 10px !important;
}

.stChatInput > div {
    background: #FFFFFF !important;
    border: 1px solid #F0DCC8 !important;
    border-radius: 14px !important;
}

.powered-badge {
    display: inline-flex;
    background: #FBE4DC;
    color: #C1502B;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.75rem;
    margin: 0 4px;
}

.hero-section {
    text-align: center;
    padding: 3rem 2rem;
}

.hero-title {
    font-size: 1.7rem;
    font-weight: 600;
    color: #4A3B32;
}

.hero-subtitle {
    color: #8B7355;
    font-size: 0.95rem;
    margin-bottom: 2rem;
}

.grid-card {
    background: #FFFFFF;
    border: 1px solid #F5DAD1;
    border-radius: 16px;
    padding: 1.4rem;
    text-align: center;
}

.grid-card-title {
    font-weight: 600;
    font-size: 1rem;
    color: #4A3B32;
}

.grid-card-desc {
    font-size: 0.82rem;
    color: #8B7355;
}

.chat-bubble-user {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 10px;
}

.chat-bubble-user .bubble {
    background: #E76F51;
    color: #FFF8F0;
    border-radius: 14px 14px 4px 14px;
    padding: 10px 14px;
    max-width: 75%;
    font-size: 14px;
}

.chat-bubble-ai {
    display: flex;
    margin-bottom: 8px;
}

.chat-bubble-ai .bubble {
    background: #FFFFFF;
    border: 1px solid #F5DAD1;
    border-radius: 14px 14px 14px 4px;
    padding: 10px 14px;
    max-width: 80%;
    font-size: 14px;
    color: #4A3B32;
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
You are a thoughtful assistant that answers questions about a YouTube video using its transcript.

Guidelines:
- Read the context carefully and answer using what is directly stated.
- If the exact detail isn't stated but can be reasonably inferred from context clues (e.g. descriptions, tone, visuals mentioned, or how someone is referred to), give your best reasoned answer and clearly say it's an inference, not a stated fact.
- Only say "This topic is not covered in the video" if there is truly no relevant information or clue to work with — do not say this just because an exact number or fact isn't spelled out.
- Give clear, conversational, well-structured answers rather than one-line dismissals.

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


def render_user_bubble(text):
    st.markdown(f"""
    <div class='chat-bubble-user'>
        <div class='bubble'>{text}</div>
    </div>
    """, unsafe_allow_html=True)


def render_ai_bubble(text):
    st.markdown(f"""
    <div class='chat-bubble-ai'>
        <div class='bubble'>{text}</div>
    </div>
    """, unsafe_allow_html=True)


# ── Header ──
st.markdown("""
<div class='main-title'>
    <h1> AskTube</h1>
    <p class='subtitle'>Drop a YouTube link. Ask anything. Get answers instantly.</p>
</div>
""", unsafe_allow_html=True)


# ── Top bar controls (replaces sidebar) ──
col1, col2 = st.columns([4, 1])
with col1:
    url = st.text_input("", placeholder="https://youtube.com/watch?v=...", label_visibility="collapsed")
with col2:
    load_btn = st.button(" Analyze")

topic = st.text_input("", placeholder="Search a topic to find timestamps...", label_visibility="collapsed", key="topic_input")
topic_btn = st.button(" Search Topic", key="topic_btn")

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
            st.success("✅ Ready to chat!")
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
            if msg["role"] == "user":
                render_user_bubble(msg["content"])
            else:
                render_ai_bubble(msg["content"])
                if msg.get("sources"):
                    with st.expander("📄 Sources"):
                        for s in msg["sources"]:
                            st.markdown(f"<div class='source-chunk'>{s}</div>", unsafe_allow_html=True)

        query = st.chat_input("Ask anything about this video...")
        if query:
            st.session_state.messages.append({"role": "user", "content": query})
            render_user_bubble(query)
            with st.spinner("Thinking..."):
                llm = get_llm()
                answer, sources = get_answer(query, st.session_state.vectorstore, llm)
                render_ai_bubble(answer)
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
                        f"<a href='{yt_link}' target='_blank' style='color:#E76F51; font-weight:600;'>▶ Jump</a>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
            else:
                st.warning(f"No mentions of **'{topic}'** found.")
        else:
            st.markdown("<p style='color:#8B7355;'>Search a topic above to find timestamps!</p>", unsafe_allow_html=True)

else:
    # Landing page
    st.markdown("""
    <div class='hero-section'>
        <p class='hero-title'>Your AI YouTube Companion</p>
        <p class='hero-subtitle'>Paste any YouTube URL and start chatting with the video instantly</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class='grid-card'>
            <p class='grid-card-title'>💬 Smart Chat</p>
            <p class='grid-card-desc'>Ask anything about the video and get accurate, detailed answers</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='grid-card'>
            <p class='grid-card-title'>📋 Auto Summary</p>
            <p class='grid-card-desc'>Get instant summary and key points without watching the full video</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='grid-card'>
            <p class='grid-card-title'>⏱️ Timestamp Search</p>
            <p class='grid-card-desc'>Find exactly when any topic is discussed with clickable timestamps</p>
        </div>""", unsafe_allow_html=True)
