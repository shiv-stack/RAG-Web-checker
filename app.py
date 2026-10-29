"""app.py
Production-ready Web Intelligence & Tech Stack Analyzer
Streamlit UI + LangChain + Google Gemini + FAISS

API key is loaded exclusively from backend environment (.env or Streamlit secrets).
Users only interact with URL input and model selection on the frontend.
"""

import os
import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from scrapers import analyze_target_website, sanitize_url
from rag_chain import (
    build_vector_store,
    create_website_rag_chain,
    format_chat_history,
    load_and_chunk_webpage,
)

# ---------------------------------------------------------
# Backend API Key Resolution (never exposed to frontend)
# ---------------------------------------------------------
def _resolve_api_key() -> str:
    """Load Gemini API key from .env or Streamlit secrets. Returns empty string if missing."""
    key = os.getenv("GOOGLE_API_KEY", "")
    if not key:
        try:
            if "GOOGLE_API_KEY" in st.secrets:
                key = str(st.secrets["GOOGLE_API_KEY"])
        except Exception:
            pass
    return key.strip()


BACKEND_API_KEY = _resolve_api_key()

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Web Intelligence & Tech Stack Analyzer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------
# Professional Design System (CSS)
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    /* ---- Global & Typography ---- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* ---- Hero Header ---- */
    .hero-container {
        text-align: center;
        padding: 2.5rem 1rem 1.2rem;
    }
    .hero-icon {
        font-size: 3rem;
        margin-bottom: 0.4rem;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #818cf8, #a78bfa, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.3rem;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #94a3b8;
        max-width: 560px;
        margin: 0 auto;
        line-height: 1.5;
    }

    /* ---- Input Area ---- */
    .input-section {
        max-width: 780px;
        margin: 1.5rem auto 2rem;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 28px 32px;
    }

    /* ---- Metric Cards ---- */
    .metric-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px 22px;
        margin-bottom: 14px;
        transition: transform 0.2s ease, border-color 0.25s ease, box-shadow 0.25s ease;
    }
    .metric-card:hover {
        border-color: rgba(139, 92, 246, 0.35);
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(139, 92, 246, 0.08);
    }
    .metric-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #64748b;
        margin-bottom: 6px;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.1rem;
        font-weight: 600;
        color: #f1f5f9;
        word-break: break-word;
    }

    /* ---- Tech Badges ---- */
    .badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 8px;
        margin-bottom: 12px;
    }
    .tech-badge {
        display: inline-flex;
        align-items: center;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.12), rgba(168, 85, 247, 0.12));
        border: 1px solid rgba(139, 92, 246, 0.28);
        color: #c4b5fd;
        padding: 6px 14px;
        border-radius: 24px;
        font-size: 0.8rem;
        font-weight: 500;
        letter-spacing: 0.01em;
        transition: background 0.2s ease, border-color 0.2s ease;
    }
    .tech-badge:hover {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.22), rgba(168, 85, 247, 0.22));
        border-color: rgba(139, 92, 246, 0.5);
    }
    .tech-badge-empty {
        color: #475569;
        font-style: italic;
        font-size: 0.85rem;
    }

    /* ---- Section Headers ---- */
    .section-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-top: 20px;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* ---- Feature Cards (Welcome) ---- */
    .feature-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 14px;
        padding: 24px 22px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.25s ease;
        height: 100%;
    }
    .feature-card:hover {
        border-color: rgba(139, 92, 246, 0.3);
        transform: translateY(-3px);
    }
    .feature-icon {
        font-size: 2.2rem;
        margin-bottom: 10px;
    }
    .feature-title {
        font-size: 1rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 8px;
    }
    .feature-desc {
        font-size: 0.85rem;
        color: #94a3b8;
        line-height: 1.55;
    }

    /* ---- Status Pill ---- */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid rgba(34, 197, 94, 0.25);
        color: #4ade80;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
    }

    /* ---- Sidebar Styling ---- */
    section[data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.95);
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    /* ---- Button overrides ---- */
    div[data-testid="stButton"] > button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    /* ---- Divider ---- */
    .styled-divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(139,92,246,0.2), transparent);
        margin: 1.5rem 0;
    }

    /* ---- Footer ---- */
    .app-footer {
        text-align: center;
        padding: 2rem 0 1rem;
        color: #475569;
        font-size: 0.78rem;
    }
    .app-footer a {
        color: #818cf8;
        text-decoration: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# State Initialization
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "analysis_data" not in st.session_state:
    st.session_state.analysis_data = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "current_url" not in st.session_state:
    st.session_state.current_url = ""

# ---------------------------------------------------------
# Hero Header
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-icon">⚡</div>
        <div class="hero-title">Web Intelligence &amp; Tech Stack Analyzer</div>
        <div class="hero-subtitle">
            Inspect live site architecture, detect frameworks &amp; CMS, and converse
            with any website using RAG‑powered AI.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Backend Key Guard
# ---------------------------------------------------------
if not BACKEND_API_KEY:
    st.error(
        "🔑 **Gemini API key not configured on the server.**\n\n"
        "The administrator must set `GOOGLE_API_KEY` in the `.env` file or Streamlit secrets "
        "before the app can function. No API key input is required from users."
    )
    st.stop()

# ---------------------------------------------------------
# Main Input Area (URL + Model + Actions)
# ---------------------------------------------------------
st.markdown('<div class="input-section">', unsafe_allow_html=True)

col_url, col_model = st.columns([3, 1])

with col_url:
    target_url = st.text_input(
        "🌐 Website URL",
        value="",
        placeholder="Enter any URL — e.g. https://stripe.com, vercel.com, github.com",
        help="Paste the full or partial URL of the website you want to analyze.",
        label_visibility="collapsed",
    )

with col_model:
    model_choice = st.selectbox(
        "Model",
        options=[
            "gemini-3.5-flash-lite",
            "gemini-3.5-flash",
            "gemini-3.8-flash",
            "gemini-pro-latest",
        ],
        index=0,
        help="Lighter models are faster and cost-effective. Pro models give deeper reasoning.",
        label_visibility="collapsed",
    )

col_analyze, col_reset, col_spacer = st.columns([1, 1, 3])
with col_analyze:
    analyze_clicked = st.button("🚀  Analyze Website", type="primary", use_container_width=True)
with col_reset:
    if st.button("🔄  Reset", use_container_width=True):
        st.session_state.messages = []
        st.session_state.analysis_data = None
        st.session_state.rag_chain = None
        st.session_state.current_url = ""
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# Pipeline Execution
# ---------------------------------------------------------
if analyze_clicked:
    if not target_url.strip():
        st.warning("⚠️ Please enter a website URL to analyze.")
    else:
        clean_target_url = sanitize_url(target_url)

        # Clear previous chat when URL changes
        if st.session_state.current_url != clean_target_url:
            st.session_state.messages = []

        with st.status("🔍 Running analysis pipeline…", expanded=True) as status:
            try:
                # Step 1
                status.write("📡 Fetching HTML headers, meta tags & WHOIS records…")
                analysis_result = analyze_target_website(clean_target_url)
                if not analysis_result.get("success"):
                    status.update(label="❌ Scraping failed", state="error")
                    st.error(f"Error scraping **{clean_target_url}**: {analysis_result.get('error')}")
                    st.stop()

                # Step 2
                status.write("✂️ Parsing & chunking web document…")
                chunks = load_and_chunk_webpage(
                    url=clean_target_url,
                    raw_html_fallback=analysis_result.get("raw_html"),
                )
                status.write(f"📄 Created **{len(chunks)} text chunks**.")

                # Step 3
                status.write("🧠 Generating vector embeddings & indexing in FAISS…")
                vectorstore = build_vector_store(chunks=chunks, api_key=BACKEND_API_KEY)

                # Step 4
                status.write("🔗 Initializing conversational RAG chain…")
                rag_chain = create_website_rag_chain(
                    vector_store=vectorstore,
                    analysis_data=analysis_result,
                    api_key=BACKEND_API_KEY,
                    model_name=model_choice,
                )

                st.session_state.analysis_data = analysis_result
                st.session_state.rag_chain = rag_chain
                st.session_state.current_url = clean_target_url

                status.update(label="✅ Analysis & RAG pipeline ready!", state="complete", expanded=False)

            except Exception as e:
                status.update(label="❌ Pipeline error", state="error")
                st.error(f"Failed to complete analysis: {str(e)}")
                st.stop()


# ---------------------------------------------------------
# Welcome State (no analysis yet)
# ---------------------------------------------------------
if not st.session_state.analysis_data:
    st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)

    col_f1, col_f2, col_f3 = st.columns(3)

    with col_f1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🛠️</div>
                <div class="feature-title">Tech Stack Detection</div>
                <div class="feature-desc">
                    Heuristic inspection of meta generators, CDN headers,
                    script sources & DOM signatures to identify frameworks,
                    CMS platforms, and analytics.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_f2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🌐</div>
                <div class="feature-title">WHOIS & Meta Intel</div>
                <div class="feature-desc">
                    Extract domain registration, registrar, country, author
                    hints, and copyright notices directly from the root
                    source in seconds.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_f3:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">💬</div>
                <div class="feature-title">AI‑Powered RAG Chat</div>
                <div class="feature-desc">
                    Ask questions about the website's content, pricing,
                    architecture, and team — answered with grounded
                    context from Gemini & FAISS.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Minimal sidebar info in welcome state
    with st.sidebar:
        st.markdown("### ⚙️ System Info")
        st.markdown(
            """
            | Component | Technology |
            |:---|:---|
            | Scraper | `requests` + `BeautifulSoup4` |
            | WHOIS | `python-whois` |
            | Embeddings | `gemini-embedding-001` |
            | Vector Store | `FAISS (faiss-cpu)` |
            | RAG Engine | `LangChain` |
            | LLM | `Google Gemini` |
            """
        )
        st.markdown("---")
        st.caption("API key is configured server-side. No key entry required.")

    st.stop()


# ---------------------------------------------------------
# Dashboard: Results
# ---------------------------------------------------------
data = st.session_state.analysis_data
meta = data.get("metadata", {})
whois_data = data.get("whois", {})
tech_stack = data.get("tech_stack", {})

st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)

# Title row with status pill
st.markdown(
    f"""
    <div style="display:flex; align-items:center; gap:14px; margin-bottom:6px;">
        <span style="font-size:1.4rem; font-weight:700; color:#f1f5f9;">
            🎯 Results for
            <a href="{data.get('url')}" target="_blank" style="color:#818cf8; text-decoration:none;">
                {data.get('domain')}
            </a>
        </span>
        <span class="status-pill">● Analyzed</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Quick metrics
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Status Code</div>
            <div class="metric-value">🟢 {meta.get('status_code', '200')} OK</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_m2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Domain Registrar</div>
            <div class="metric-value">{whois_data.get('registrar', 'N/A')[:28]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_m3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Created Date</div>
            <div class="metric-value">{whois_data.get('creation_date', 'N/A')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_m4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Origin Country / Org</div>
            <div class="metric-value">{whois_data.get('country', 'N/A')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Metadata expander
with st.expander("📋 Page Metadata & Creator Details", expanded=True):
    col_meta1, col_meta2 = st.columns([2, 1])
    with col_meta1:
        st.markdown(f"**Title:** {meta.get('title', 'N/A')}")
        st.markdown(f"**Description:** {meta.get('description', 'N/A')}")
    with col_meta2:
        st.markdown(f"**Author / Creator:** {meta.get('author', 'Not specified')}")
        st.markdown(f"**Copyright:** `{meta.get('copyright', 'None detected')}`")
        if whois_data.get("expiration_date") != "Not Available":
            st.markdown(f"**Domain Expiry:** {whois_data.get('expiration_date')}")

# Tech Stack
st.markdown('<div class="section-title">🧩 Detected Technology Stack</div>', unsafe_allow_html=True)

col_t1, col_t2, col_t3 = st.columns(3)


def render_badge_list(category_name: str, badges: list):
    html = f"<strong>{category_name}</strong><div class='badge-container'>"
    if badges:
        for badge in badges:
            html += f"<span class='tech-badge'>{badge}</span>"
    else:
        html += "<span class='tech-badge-empty'>None directly detected</span>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


with col_t1:
    render_badge_list("CMS & Platforms", tech_stack.get("CMS & Platforms", []))
    render_badge_list("Frontend Frameworks", tech_stack.get("Frontend Frameworks", []))

with col_t2:
    render_badge_list("Styling & UI", tech_stack.get("Styling & UI", []))
    render_badge_list("Hosting & CDN", tech_stack.get("Hosting & CDN", []))

with col_t3:
    render_badge_list("Analytics & Tags", tech_stack.get("Analytics & Tags", []))
    render_badge_list("Server & Backend", tech_stack.get("Server & Backend", []))

st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)

# ---------------------------------------------------------
# RAG Chat Interface
# ---------------------------------------------------------
st.markdown('<div class="section-title">💬 Website Q&A — AI Chat</div>', unsafe_allow_html=True)
st.caption("Ask anything about the website's content, offerings, architecture, or mission.")

# Preset questions
col_p1, col_p2, col_p3, col_p4 = st.columns(4)
preset_question = None
with col_p1:
    if st.button("💡 What does this site do?", use_container_width=True):
        preset_question = "What does this business or website do? Summarize its key purpose."
with col_p2:
    if st.button("🛠️ Explain the tech stack", use_container_width=True):
        preset_question = "What tech stack powers this website based on the analysis and content?"
with col_p3:
    if st.button("👤 Who created this site?", use_container_width=True):
        preset_question = "Who is the author, creator, or organization behind this website?"
with col_p4:
    if st.button("💰 Features & pricing", use_container_width=True):
        preset_question = "What are the core features, offerings, or pricing mentioned on this website?"

# Render past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Process input
user_prompt = st.chat_input("Ask a question about this website…") or preset_question

if user_prompt:
    if not st.session_state.rag_chain:
        st.warning("⚠️ RAG Chain not initialized. Please click **Analyze Website** first.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Analyzing context & generating response…"):
                try:
                    langchain_history = format_chat_history(st.session_state.messages[:-1])
                    response = st.session_state.rag_chain.invoke({
                        "input": user_prompt,
                        "chat_history": langchain_history,
                    })
                    answer = response.get("answer", "No response generated.")
                    message_placeholder.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as err:
                    err_msg = f"❌ Error querying RAG chain: {str(err)}"
                    message_placeholder.markdown(err_msg)
                    st.session_state.messages.append({"role": "assistant", "content": err_msg})

# ---------------------------------------------------------
# Sidebar (when analysis is active)
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Active Session")
    st.markdown(f"**URL:** `{st.session_state.current_url}`")
    st.markdown(f"**Model:** `{model_choice}`")
    st.markdown("---")
    st.markdown(
        """
        | Component | Technology |
        |:---|:---|
        | Scraper | `requests` + `BS4` |
        | WHOIS | `python-whois` |
        | Embeddings | `gemini-embedding-001` |
        | Vector Store | `FAISS` |
        | RAG | `LangChain` |
        | LLM | `Google Gemini` |
        """
    )
    st.markdown("---")
    st.caption("🔒 API key is managed server-side.")

# Footer
st.markdown(
    """
    <div class="app-footer">
        Built with Streamlit · LangChain · Google Gemini · FAISS<br/>
        <span style="color:#64748b;">API key secured on the backend — no user input required.</span>
    </div>
    """,
    unsafe_allow_html=True,
)
