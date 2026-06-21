import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss
import numpy as np
import google.generativeai as genai

# -------------------------------
# Gemini Configuration
# -------------------------------

st.set_page_config(
    page_title="AI Research Paper Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


GEMINI_API_KEY = ""
genai.configure(api_key=GEMINI_API_KEY)

# -------------------------------
# Load Embedding Model
# -------------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

# -------------------------------
# Custom CSS
# -------------------------------

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .stMetric {
        background-color: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 15px;
    }
    .chat-question {
        background-color: #eff6ff;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 8px;
        border-left: 4px solid #3b82f6;
    }
    .chat-answer {
        background-color: #f9fafb;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 16px;
        border-left: 4px solid #10b981;
    }
    .paper-card {
        background-color: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 10px 15px;
        margin-bottom: 8px;
    }
    div.stButton > button {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
    }
    .section-divider {
        margin: 2rem 0;
        border-top: 1px solid #e5e7eb;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Header
# -------------------------------

st.markdown('<p class="main-header">📚 AI Research Paper Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload, analyze, summarize and chat with your research papers using AI</p>', unsafe_allow_html=True)

# -------------------------------
# Sidebar
# -------------------------------

with st.sidebar:
    st.markdown("## 🧭 About")
    st.info(
        """
        **Features**
        - 📄 Multi-PDF Upload
        - 💬 Research Q&A (Chat)
        - 🔍 Semantic Search (FAISS)
        - ✨ Gemini AI Responses
        - 📝 Research Summaries
        - 🏷️ Keyword Extraction
        - ⚖️ Paper Comparison
        """
    )
    st.markdown("---")
    st.markdown("### ⚙️ How it works")
    st.markdown(
        """
        1. Upload one or more PDF research papers  
        2. Wait for embeddings to be generated  
        3. Use the tools below to summarize, extract keywords, compare, or ask questions  
        """
    )
    st.markdown("---")
    st.caption("Powered by Gemini 2.5 Flash + FAISS + Sentence Transformers")

# -------------------------------
# Upload PDFs
# -------------------------------

uploaded_files = st.file_uploader(
    "📤 Upload Research Papers (PDF)",
    type=["pdf"],
    accept_multiple_files=True,
    help="You can upload multiple PDF files at once"
)

# -------------------------------
# Process PDFs
# -------------------------------

if uploaded_files:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200 )

    papers = {}
    chunks = []
    chunk_metadata = []
    combined_text = ""

    with st.expander(f"📄 Uploaded Papers ({len(uploaded_files)})", expanded=True):
        for uploaded_file in uploaded_files:
            paper_text = ""
            pdf = PdfReader(uploaded_file)

            for page_num, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text()

                if page_text:
                    paper_text += page_text + "\n"

                    page_chunks = splitter.split_text(page_text)

                    for chunk in page_chunks:
                        chunks.append(chunk)

                        chunk_metadata.append({
                            "paper": uploaded_file.name,
                            "page": page_num,
                            "text": chunk
                        })

            papers[uploaded_file.name] = paper_text
            combined_text += paper_text + "\n"

    # -------------------------------
    # Embeddings + FAISS Index
    # -------------------------------

    with st.spinner("🔄 Creating embeddings and building search index..."):
        embeddings = embedding_model.encode(chunks)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype("float32"))

    st.success("✅ Research papers processed successfully!")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📚 Documents Uploaded", len(uploaded_files))
    with col2:
        st.metric("🧩 Text Chunks", len(chunks))
    with col3:
        st.metric("📐 Embedding Dimension", dimension)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # -------------------------------
    # Tabs for main features
    # -------------------------------

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📝 Summary",
        "🏷️ Keywords",
        "⚖️ Compare Papers",
        "💬 Ask Questions",
        "📈 Research Insights"
    ])

    # -------------------------------
    # Summary Tab
    # -------------------------------

    with tab1:
        st.markdown("### Generate Research Summary")
        st.caption("Get an overview of objective, methodology, findings, limitations, and future work.")

        if st.button("✨ Generate Summary", key="summary_btn"):
            with st.spinner("Generating summary..."):
                model = genai.GenerativeModel("gemini-2.5-flash")

                summary_prompt = f"""
                Analyze the uploaded research papers.

                Provide:
                1. Main Objective
                2. Methodology
                3. Key Findings
                4. Limitations
                5. Future Work

                Content:
                {combined_text[:15000]}
                """

                response = model.generate_content(summary_prompt)

                st.markdown("#### 📋 Research Summary")
                st.markdown(response.text)
                st.download_button(
                label="⬇ Download Summary",
                data=response.text,
                file_name="research_summary.txt",
                mime="text/plain"
                )

    # -------------------------------
    # Keywords Tab
    # -------------------------------

    with tab2:
        st.markdown("### Extract Keywords & Concepts")
        st.caption("Identify key terms, technologies, models, and datasets used in the papers.")

        if st.button("🔍 Extract Keywords", key="keywords_btn"):
            with st.spinner("Extracting keywords..."):
                model = genai.GenerativeModel("gemini-2.5-flash")

                keyword_prompt = f"""
                Analyze the uploaded research papers.

                Extract:
                1. Important Keywords
                2. Technologies Used
                3. Models Mentioned
                4. Datasets Used
                5. Research Areas

                Content:
                {combined_text[:15000]}
                """

                response = model.generate_content(keyword_prompt)

                st.markdown("#### 🏷️ Extracted Keywords")
                st.markdown(response.text)

    # -------------------------------
    # Compare Papers Tab
    # -------------------------------

    with tab3:
        st.markdown("### Compare Research Papers")

        paper_names = list(papers.keys())

        if len(paper_names) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                paper1 = st.selectbox("📄 Select First Paper", paper_names, index=0)
            with col2:
                paper2 = st.selectbox("📄 Select Second Paper", paper_names, index=1)

            if st.button("⚖️ Compare Selected Papers", key="compare_btn"):
                with st.spinner("Comparing papers..."):
                    model = genai.GenerativeModel("gemini-2.5-flash")

                    compare_prompt = f"""
                    You are an academic research analyst.

                    Compare the following two research papers.

                    PAPER 1 TITLE:
                    {paper1}

                    PAPER 1 CONTENT:
                    {papers[paper1][:5000]}

                    PAPER 2 TITLE:
                    {paper2}

                    PAPER 2 CONTENT:
                    {papers[paper2][:5000]}

                    Create a clean markdown table with exactly 3 columns:

                    | Aspect | {paper1} | {paper2} |

                    Compare the following aspects:
                    1. Research Objective
                    2. Methodology
                    3. Dataset Used
                    4. Model/Technology Used
                    5. Key Results
                    6. Advantages
                    7. Limitations
                    8. Future Work

                    Rules:
                    - Keep each cell concise.
                    - Do not repeat paper titles inside cells.
                    - Use proper markdown table formatting.
                    - Do not generate nested tables.
                    - Do not repeat information.
                    - If information is unavailable, write "Not Specified".

                    Return ONLY the comparison table.
                    """

                    response = model.generate_content(compare_prompt)

                    st.markdown("#### 📊 Comparison Result")
                    st.markdown(response.text)
        else:
            st.info("ℹ️ Upload at least 2 papers to use the comparison feature.")

    # -------------------------------
    # Question Answering Tab
    # -------------------------------

    with tab4:
        st.markdown("### Ask Questions About Your Papers")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        question = st.text_input("💬 Enter your question", placeholder="e.g. What dataset was used in Paper X?")

        col_ask, col_clear = st.columns([1, 1])
        with col_ask:
            ask_clicked = st.button("🚀 Ask", key="ask_btn")
        with col_clear:
            if st.button("🗑️ Clear Chat History", key="clear_btn"):
                st.session_state.chat_history = []
                st.rerun()

        if ask_clicked and question:
            with st.spinner("Thinking..."):
                query_embedding = embedding_model.encode([question])

                distances, indices = index.search(
                    np.array(query_embedding).astype("float32"),
                    k=10
                )

                context = ""
                for idx in indices[0]:
                    context += chunks[idx] + "\n\n"

                prompt = f"""
                You are an expert AI research assistant analyzing multiple uploaded research papers.

                If the question asks for comparison, summarize information from all relevant papers found in the context.
                Mention paper names whenever possible.
                If the answer is not available in the context, say:
                "The answer is not available in the uploaded research papers."

                Context:
                {context}

                Question:
                {question}

                Answer:
                """

                model = genai.GenerativeModel("gemini-2.5-flash")
                response = model.generate_content(prompt)

                st.session_state.chat_history.append({
                    "question": question,
                    "answer": response.text
                })
    # -------------------------------
    # Research Insights Tab
    # -------------------------------

    with tab5:

        st.markdown("### 📈 Research Insights")
        st.caption(
            "Identify research trends, datasets, models, gaps, and future opportunities."
        )

        if st.button("📊 Generate Insights", key="insights_btn"):

            with st.spinner("Analyzing research papers..."):

                try:

                    model = genai.GenerativeModel("gemini-2.5-flash")

                    insights_prompt = f"""
                    Analyze all uploaded research papers.

                    Provide:

                    1. Common Research Themes
                    2. Frequently Used Datasets
                    3. Frequently Used Models / Algorithms
                    4. Common Evaluation Metrics
                    5. Research Gaps Identified
                    6. Future Research Opportunities
                    7. Emerging Trends

                    Present the results in a structured format.

                    Content:

                    {combined_text[:15000]}
                    """

                    response = model.generate_content(insights_prompt)

                    st.markdown("#### 📈 Research Insights")
                    st.markdown(response.text)

                except Exception as e:

                    st.error(f"Error generating insights: {e}")

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        if st.session_state.chat_history:
            st.markdown("#### 💬 Chat History")

            for chat in reversed(st.session_state.chat_history):
                st.markdown(
                    f'<div class="chat-question">🙋 <b>Q:</b> {chat["question"]}</div>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'<div class="chat-answer">🤖 <b>A:</b> {chat["answer"]}</div>',
                    unsafe_allow_html=True
                )
        else:
            st.caption("No questions asked yet. Start by typing a question above!")

else:
    st.markdown(
        """
        <div style="text-align:center; padding: 60px 20px; color:#9ca3af;">
            <h3>👋 Welcome!</h3>
            <p>Upload one or more research papers (PDF) above to get started.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
