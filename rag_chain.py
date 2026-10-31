"""rag_chain.py
LangChain Vector Store and Retrieval-Augmented Generation (RAG) pipeline
powered by Google Gemini and FAISS.
"""

import ast
from typing import Any, Dict, List, Optional
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


CUSTOM_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    )
}


def load_and_chunk_webpage(url: str, raw_html_fallback: Optional[str] = None) -> List[Document]:
    """
    Parse text content from raw HTML fallback (to avoid redundant network requests),
    or fall back to WebBaseLoader.
    Chunks text with RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).
    """
    documents: List[Document] = []

    # Prefer raw_html_fallback to avoid making a redundant outbound request
    if raw_html_fallback and len(raw_html_fallback) > 100:
        try:
            soup = BeautifulSoup(raw_html_fallback, "html.parser")
            for tag in soup(["script", "style", "noscript", "svg", "nav"]):
                tag.decompose()
            extracted_text = soup.get_text(separator="\n", strip=True)
            if extracted_text and len(extracted_text) >= 100:
                documents = [Document(page_content=extracted_text, metadata={"source": url})]
        except Exception:
            pass

    # If raw_html_fallback wasn't available or yielded sparse text, use WebBaseLoader
    if not documents:
        try:
            loader = WebBaseLoader(
                web_paths=(url,),
                header_template=CUSTOM_HEADERS,
                verify_ssl=False,
            )
            documents = loader.load()
        except Exception:
            pass

    if not documents or not any(d.page_content.strip() for d in documents):
        raise RuntimeError(f"Unable to extract readable text content from {url}.")

    # Clean text whitespace in documents
    for doc in documents:
        lines = [line.strip() for line in doc.page_content.splitlines() if line.strip()]
        doc.page_content = "\n".join(lines)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


def build_vector_store(
    chunks: List[Document],
    api_key: str,
    embedding_model: str = "models/gemini-embedding-001",
) -> FAISS:
    """
    Generate embeddings using Google Gemini and index chunks into an in-memory FAISS store.
    Tries valid Google Gemini embedding models gracefully to ensure 100% compatibility.
    """
    if not chunks:
        raise ValueError("No text chunks available to build vector store.")

    candidate_models = [
        embedding_model,
        "models/gemini-embedding-001",
        "models/gemini-embedding-2-preview",
        "models/gemini-embedding-2",
        "models/text-embedding-004",
    ]
    # Deduplicate while preserving order
    candidate_models = list(dict.fromkeys(candidate_models))

    last_error = None
    for model_name in candidate_models:
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model=model_name,
                google_api_key=api_key,
            )
            vector_store = FAISS.from_documents(chunks, embeddings)
            return vector_store
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(f"Error generating embeddings with Gemini models: {str(last_error)}")


def format_tech_context(analysis_data: Dict[str, Any]) -> str:
    """
    Format structured technical metadata (WHOIS, tech stack, meta tags)
    into a structured string for prompt grounding.
    """
    meta = analysis_data.get("metadata", {})
    whois = analysis_data.get("whois", {})
    tech = analysis_data.get("tech_stack", {})

    context_lines = [
        "### STRUCTURED TECHNICAL METADATA FOR THIS WEBSITE:",
        f"- Target URL: {analysis_data.get('url', 'N/A')}",
        f"- Root Domain: {analysis_data.get('domain', 'N/A')}",
        f"- Meta Title: {meta.get('title', 'N/A')}",
        f"- Meta Description: {meta.get('description', 'N/A')}",
        f"- Author / Creator Hint: {meta.get('author', 'N/A')}",
        f"- Copyright / Footer Hint: {meta.get('copyright', 'N/A')}",
        f"- Domain Registrar: {whois.get('registrar', 'N/A')}",
        f"- Domain Created Date: {whois.get('creation_date', 'N/A')}",
        f"- Domain Expiration Date: {whois.get('expiration_date', 'N/A')}",
        f"- Domain Country / Org: {whois.get('country', 'N/A')} / {whois.get('org', 'N/A')}",
        "",
        "### DETECTED TECH STACK BY CATEGORY:",
    ]

    for category, items in tech.items():
        val = ", ".join(items) if items else "None directly detected"
        context_lines.append(f"- {category}: {val}")

    return "\n".join(context_lines)


def extract_text_from_llm_response(content: Any) -> str:
    """
    Safely extract clean markdown/plain text from an LLM response.
    Handles string output, multimodal list of content blocks (e.g. [{'type': 'text', 'text': '...'}]),
    stringified data structures, and dictionary structures returned by newer LangChain Google GenAI versions.
    """
    if isinstance(content, str):
        trimmed = content.strip()
        # If the string is a stringified list or dict, parse it
        if (trimmed.startswith("[{") and trimmed.endswith("}]")) or (trimmed.startswith("{'") and trimmed.endswith("'}")):
            try:
                parsed = ast.literal_eval(trimmed)
                return extract_text_from_llm_response(parsed)
            except Exception:
                pass
        return trimmed

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                # Check for standard content block keys: 'text', 'content'
                if "text" in block and block["text"]:
                    parts.append(str(block["text"]))
                elif "content" in block and block["content"]:
                    parts.append(str(block["content"]))
            elif hasattr(block, "text") and block.text:
                parts.append(str(block.text))
        if parts:
            return "\n".join(parts).strip()

    if isinstance(content, dict):
        if "text" in content and content["text"]:
            return str(content["text"]).strip()
        if "content" in content and content["content"]:
            return str(content["content"]).strip()

    return str(content).strip()


class WebsiteRAG:
    """Quota-efficient, conversational RAG engine for website analysis."""

    def __init__(
        self,
        vector_store: FAISS,
        analysis_data: Dict[str, Any],
        api_key: str,
        model_name: str = "gemini-3.5-flash-lite",
        temperature: float = 0.2,
    ):
        self.vector_store = vector_store
        self.analysis_data = analysis_data
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature,
        )
        self.retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4},
        )
        self.tech_context = format_tech_context(analysis_data)

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        user_input = inputs.get("input", "")
        chat_history = inputs.get("chat_history", [])

        # Retrieve relevant chunks directly (1 local FAISS vector search, 0 LLM calls)
        docs = self.retriever.invoke(user_input)
        retrieved_context = "\n\n---\n\n".join([d.page_content for d in docs])

        system_prompt = (
            "You are an elite Web Intelligence & Tech Stack Analyzer assistant.\n"
            "You have access to two sources of truth:\n"
            "1. Extracted Technical & WHOIS Metadata (provided below).\n"
            "2. Retrieved Web Content chunks from the scraped website.\n\n"
            f"{self.tech_context}\n\n"
            "GUIDELINES:\n"
            "- Prioritize technical metadata when answering questions about tech stack, CMS, frameworks, hosting, WHOIS, or authors.\n"
            "- Prioritize retrieved web content when answering questions about product offerings, features, pricing, or mission.\n"
            "- Be concise, professional, and factual. Format technologies, libraries, and findings clearly with bullet points or bold text.\n"
            "- If the answer cannot be found in either the metadata or retrieved content, state clearly: "
            "'This information could not be found on the analyzed website.' Do not hallucinate or guess.\n\n"
            "RETRIEVED WEB CONTENT:\n"
            f"{retrieved_context}"
        )

        messages = [SystemMessage(content=system_prompt)]
        for msg in chat_history[-6:]:
            if isinstance(msg, HumanMessage):
                messages.append(msg)
            elif isinstance(msg, AIMessage):
                messages.append(msg)
        messages.append(HumanMessage(content=user_input))

        try:
            response = self.llm.invoke(messages)
            clean_answer = extract_text_from_llm_response(response.content)
            return {"answer": clean_answer}
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "quota" in err_str.lower() or "ResourceExhausted" in err_str:
                return {
                    "answer": (
                        "⏳ **Google Gemini Free Tier Rate Limit Exceeded (5 requests/minute).**\n\n"
                        "Google's free tier allows up to 5 requests per minute for this project. "
                        "Please wait ~15-30 seconds before asking the next question, or choose a lighter model "
                        "(like `gemini-3.5-flash-lite`) in the sidebar."
                    )
                }
            raise e


def create_website_rag_chain(
    vector_store: FAISS,
    analysis_data: Dict[str, Any],
    api_key: str,
    model_name: str = "gemini-3.5-flash-lite",
    temperature: float = 0.2,
) -> WebsiteRAG:
    """Create and return a configured WebsiteRAG instance."""
    return WebsiteRAG(
        vector_store=vector_store,
        analysis_data=analysis_data,
        api_key=api_key,
        model_name=model_name,
        temperature=temperature,
    )


def format_chat_history(messages: List[Dict[str, str]]) -> List[Any]:
    """Convert session state message dictionaries to LangChain Human/AIMessage objects."""
    formatted = []
    for msg in messages:
        if msg["role"] == "user":
            formatted.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            formatted.append(AIMessage(content=msg["content"]))
    return formatted
