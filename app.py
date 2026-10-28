"""app.py
Production-ready Web Intelligence & Tech Stack Analyzer
Streamlit UI + LangChain + Google Gemini + FAISS
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
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Web Intelligence & Tech Stack Analyzer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Custom Styling (Glassmorphism & Modern Palette)
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    /* Metric & Card styling */
    .metric-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 14px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        transform: translateY(-2px);
    }
    .metric-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.15rem;
        font-weight: 600;
        color: #f8fafc;
        word-break: break-word;
    }
    /* Tech Badges */
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
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(168, 85, 247, 0.15));
        border: 1px solid rgba(139, 92, 246, 0.35);
        color: #c4b5fd;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 500;
        letter-spacing: 0.02em;
    }
    .tech-badge-empty {
        color: #64748b;
        font-style: italic;
        font-size: 0.85rem;
    }
    /* Section headers */
    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-top: 18px;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    /* Quick prompt button style */
    div[data-testid="stButton"] > button {
        border-radius: 8px;
        transition: all 0.2s ease;
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
# Sidebar Configuration
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ Configuration")
    st.markdown("Analyze any live website architecture, tech stack & interact via Gemini RAG.")

    # 1. API Key detection from environment, secrets, or manual entry
    env_api_key = os.getenv("GOOGLE_API_KEY", "")
    if not env_api_key:
        try:
            if "GOOGLE_API_KEY" in st.secrets:
                env_api_key = str(st.secrets["GOOGLE_API_KEY"])
        except Exception:
            pass

    api_key_input = st.text_input(
        "Google Gemini API Key",
        value=env_api_key,
        type="password",
        placeholder="AIzaSy...",
        help="Reads from GOOGLE_API_KEY (.env or Streamlit secrets) if set. Required for vector embeddings & chat.",
    )
    active_api_key = api_key_input.strip() or env_api_key

    # 2. Target Website URL
    target_url = st.text_input(
        "Target Website URL",
        value="https://supabase.com",
        placeholder="https://example.com",
        help="Enter the complete or partial URL (e.g., vercel.com or https://stripe.com)",
    )

    # 3. Model Selection
    model_choice = st.selectbox(
        "Gemini LLM Model",
        options=["gemini-3.5-flash-lite", "gemini-flash-latest", "gemini-3.8-flash", "gemini-pro-latest"],
        index=0,
        help="Gemini 3.5 Flash Lite is optimal for free-tier quotas and high speed. Pro models provide deeper technical reasoning.",
    )

    # 4. Action Buttons
    col_run, col_clear = st.columns([1, 1])
    with col_run:
        analyze_clicked = st.button("🚀 Analyze", type="primary", use_container_width=True)
    with col_clear:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.messages = []
            st.session_state.analysis_data = None
            st.session_state.rag_chain = None
            st.session_state.current_url = ""
            st.rerun()

    st.markdown("---")
    st.markdown(
        """
        **System Components:**
        - **Scraper**: `requests` + `BeautifulSoup4`
        - **WHOIS**: `python-whois`
        - **Embeddings**: `models/embedding-001`
        - **Vector Store**: `FAISS (faiss-cpu)`
        - **RAG Engine**: `LangChain`
        """
    )


# ---------------------------------------------------------
# Execution Logic (Pipeline Run)
# ---------------------------------------------------------
if analyze_clicked:
    if not active_api_key:
        st.error("⚠️ Please provide a valid **Google Gemini API Key** in the sidebar or via the `GOOGLE_API_KEY` environment variable.")
    elif not target_url.strip():
        st.warning("⚠️ Please provide a target URL to analyze.")
    else:
        clean_target_url = sanitize_url(target_url)

        # Clear previous chat session on new URL analysis
        if st.session_state.current_url != clean_target_url:
            st.session_state.messages = []

        with st.status("🔍 Analyzing website intelligence pipeline...", expanded=True) as status:
            try:
                # Step 1: Scrape & Detect
                status.write("📡 Fetching HTML headers, meta tags, and WHOIS records...")
                analysis_result = analyze_target_website(clean_target_url)
                if not analysis_result.get("success"):
                    status.update(label="❌ Analysis failed during scraping", state="error")
                    st.error(f"Error scraping {clean_target_url}: {analysis_result.get('error')}")
                    st.stop()

                # Step 2: Chunk Page Content
                status.write("✂️ Parsing and chunking web document with `RecursiveCharacterTextSplitter`...")
                chunks = load_and_chunk_webpage(
                    url=clean_target_url,
                    raw_html_fallback=analysis_result.get("raw_html"),
                )
                status.write(f"📄 Successfully created **{len(chunks)} text chunks**.")

                # Step 3: Embeddings & Vector Store
                status.write("🧠 Generating vector embeddings with Google Gemini & indexing in FAISS...")
                vectorstore = build_vector_store(chunks=chunks, api_key=active_api_key)

                # Step 4: Build RAG Chain
                status.write("🔗 Initializing conversational RAG retrieval chain...")
                rag_chain = create_website_rag_chain(
                    vector_store=vectorstore,
                    analysis_data=analysis_result,
                    api_key=active_api_key,
                    model_name=model_choice,
                )

                # Save into state
                st.session_state.analysis_data = analysis_result
                st.session_state.rag_chain = rag_chain
                st.session_state.current_url = clean_target_url

                status.update(label="✅ Analysis & RAG Pipeline Ready!", state="complete", expanded=False)

            except Exception as e:
                status.update(label="❌ Pipeline execution error", state="error")
                st.error(f"Failed to complete analysis pipeline: {str(e)}")
                st.stop()


# ---------------------------------------------------------
# Main Panel View
# ---------------------------------------------------------
st.title("⚡ Web Intelligence & Tech Stack Analyzer")
st.caption("Inspect live site architecture, detect frameworks & CMS, and converse with the website via RAG.")

# Welcome State
if not st.session_state.analysis_data:
    st.info("👈 Enter your Gemini API Key and a Target Website URL in the sidebar, then click **Analyze** to begin.")
    
    col_feat1, col_feat2, col_feat3 = st.columns(3)
    with col_feat1:
        st.markdown(
            """
            #### 🛠️ Tech Stack Detector
            Heuristic inspection of meta generators, scripts, CDN headers, and DOM signatures to reveal frameworks, styling libraries, and analytics.
            """
        )
    with col_feat2:
        st.markdown(
            """
            #### 🌐 WHOIS & Meta Intel
            Extract domain registration dates, registrar, country, author hints, and copyright notices directly from root sources.
            """
        )
    with col_feat3:
        st.markdown(
            """
            #### 💬 Grounded RAG Chat
            Ask direct questions about business models, pricing, architecture, and team offerings powered by Gemini & FAISS vector search.
            """
        )
    st.stop()

# ---------------------------------------------------------
# Dashboard: Metadata & WHOIS & Tech Stack
# ---------------------------------------------------------
data = st.session_state.analysis_data
meta = data.get("metadata", {})
whois_data = data.get("whois", {})
tech_stack = data.get("tech_stack", {})

# Quick summary bar
st.markdown(f"### 🎯 Results for [{data.get('domain')}]({data.get('url')})")

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
            <div class="metric-value">{whois_data.get('registrar', 'N/A')[:22]}</div>
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

# Metadata Details
with st.expander("📋 View Page Metadata & Creator Details", expanded=True):
    col_meta1, col_meta2 = st.columns([2, 1])
    with col_meta1:
        st.markdown(f"**Title:** {meta.get('title', 'N/A')}")
        st.markdown(f"**Meta Description:** {meta.get('description', 'N/A')}")
    with col_meta2:
        st.markdown(f"**Author / Creator:** {meta.get('author', 'Not specified')}")
        st.markdown(f"**Copyright Notice:** `{meta.get('copyright', 'None detected')}`")
        if whois_data.get("expiration_date") != "Not Available":
            st.markdown(f"**Domain Expiry:** {whois_data.get('expiration_date')}")

# Detected Tech Stack Cards
st.markdown("<div class='section-title'>🧩 Detected Technology Stack</div>", unsafe_allow_html=True)

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

st.markdown("---")

# ---------------------------------------------------------
# RAG Chat Interface
# ---------------------------------------------------------
st.markdown("<div class='section-title'>💬 Interactive Website Q&A (RAG)</div>", unsafe_allow_html=True)
st.caption("Ask anything about this website's content, offerings, architecture, or mission.")

# Prompt shortcut buttons
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
    if st.button("💰 Key features & pricing", use_container_width=True):
        preset_question = "What are the core features, offerings, or pricing mentioned on this website?"

# Render past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Process User Input (either from chat_input or preset button)
user_prompt = st.chat_input("Ask a question about this website...") or preset_question

if user_prompt:
    if not active_api_key:
        st.error("⚠️ Google Gemini API Key is missing. Please set it in the sidebar.")
    elif not st.session_state.rag_chain:
        st.warning("⚠️ RAG Chain is not initialized. Please click 'Analyze' first.")
    else:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Generate Assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Analyzing document context & generating response..."):
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
