
import os
from datetime import datetime

import requests
import streamlit as st
from dotenv import load_dotenv


load_dotenv()


st.set_page_config(
    page_title="Research Gap Finder",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


API_URL = os.getenv("STREAMLIT_API_URL", "http://localhost:5000")


def analyze_topic(topic: str, language: str = "English") -> dict:
    """
    Call the backend API to analyze a research topic.

    Args:
        topic: The research topic to analyze
        language: Preferred language for generated output

    Returns:
        Dictionary containing analysis results

    Raises:
        requests.HTTPError: If the API call fails
    """
    response = requests.post(
        f"{API_URL}/analyze",
        json={"topic": topic, "language": language},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def display_gap_score(gap_score: str):
    """
    Display the gap score in a prominent way.

    Args:
        gap_score: The gap score to display (e.g., "7.5/10")
    """
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            f"""
            <div class="gap-score-card">
                <p class="gap-score-label">GAP SCORE</p>
                <p class="gap-score-value">{gap_score}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def display_results_group(title: str, items: list, icon: str = "📋"):
    """
    Display a group of results (trends, gaps, ideas, etc.).

    Args:
        title: The title of the group
        items: List of items to display
        icon: Emoji icon to display
    """
    if not items:
        return

    st.markdown(
        f'<div class="result-group-title">{icon} {title}</div>',
        unsafe_allow_html=True,
    )

    for i, item in enumerate(items, 1):
        st.markdown(
            f"""
            <div class="result-card" style="animation-delay: {i * 0.08}s;">
                <span class="result-index">{i}.</span>
                <span class="result-text"> {item}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def format_history_as_text(history: list) -> str:
    """
    Convert search history into a downloadable text document.

    Args:
        history: List of history entries

    Returns:
        Formatted plain text for download
    """
    lines = [
        "Research Gap Finder - Search History",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    if not history:
        lines.append("No search history available.")
        return "\n".join(lines)

    for i, entry in enumerate(history, 1):
        lines.append(f"{i}. Timestamp: {entry.get('timestamp', 'N/A')}")
        lines.append(f"Topic: {entry.get('topic', 'N/A')}")
        lines.append(f"Language: {entry.get('language', 'N/A')}")
        lines.append(f"Gap Score: {entry.get('gap_score', 'N/A')}")

        for section_title, key in [
            ("Current Trends", "trends"),
            ("Research Gaps", "gaps"),
            ("Project Ideas", "ideas"),
            ("Future Directions", "future_work"),
        ]:
            lines.append(f"{section_title}:")
            items = entry.get(key, [])
            if items:
                for idx, item in enumerate(items, 1):
                    lines.append(f"  {idx}. {item}")
            else:
                lines.append("  - None")

        lines.append("")

    return "\n".join(lines)


def save_search_to_history(topic: str, language: str, results: dict):
    """
    Save successful search results to session history.

    Args:
        topic: User query topic
        language: Preferred output language
        results: API result payload
    """
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "topic": topic.strip(),
        "language": language.strip(),
        "gap_score": results.get("gap_score", "N/A"),
        "trends": results.get("trends", []),
        "gaps": results.get("gaps", []),
        "ideas": results.get("ideas", []),
        "future_work": results.get("future_work", []),
        "model_used": results.get("model_used", "Unknown"),
    }
    st.session_state.search_history.insert(0, entry)
    st.session_state.latest_results = results


def display_analysis_results(results: dict):
    """
    Display complete analysis output from backend results.

    Args:
        results: API result payload
    """
    gap_score = results.get("gap_score", "N/A")
    display_gap_score(gap_score)

    col_left, col_right = st.columns(2)

    with col_left:
        display_results_group("Current Trends", results.get("trends", []), icon="📈")
        display_results_group("Research Gaps", results.get("gaps", []), icon="🎯")

    with col_right:
        display_results_group("Project Ideas", results.get("ideas", []), icon="💡")
        display_results_group("Future Directions", results.get("future_work", []), icon="🔮")

    st.markdown("---")
    st.caption(
        f"✨ Powered by: {results.get('model_used', 'Unknown')} | Language: {results.get('language_used', 'English')}"
    )


def main():
    """
    Main function to run the Streamlit app.
    """
    if "search_history" not in st.session_state:
        st.session_state.search_history = []
    if "latest_results" not in st.session_state:
        st.session_state.latest_results = None

    st.markdown(
        """
        <style>
        :root {
            --bg-0: #020617;
            --bg-1: #0f172a;
            --glass: rgba(255, 255, 255, 0.06);
            --glass-border: rgba(255, 255, 255, 0.17);
            --text-main: #e2e8f0;
            --text-muted: #94a3b8;
            --accent-a: #22d3ee;
            --accent-b: #3b82f6;
            --accent-c: #34d399;
            --card-text: #e2e8f0;
        }
        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 12% 15%, rgba(56, 189, 248, 0.20), transparent 30%),
                radial-gradient(circle at 85% 10%, rgba(96, 165, 250, 0.15), transparent 32%),
                radial-gradient(circle at 50% 78%, rgba(52, 211, 153, 0.12), transparent 30%),
                linear-gradient(155deg, var(--bg-0) 0%, var(--bg-1) 55%, #020b1f 100%);
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(10, 20, 45, 0.92), rgba(2, 6, 23, 0.92));
            border-right: 1px solid var(--glass-border);
            backdrop-filter: blur(12px);
        }
        [data-testid="stSidebar"] * {
            color: var(--text-main);
        }
        [data-testid="stSidebar"] .stCaption {
            color: var(--text-muted);
        }
        [data-testid="stHeader"] {
            background: rgba(2, 6, 23, 0.35);
            backdrop-filter: blur(10px);
        }
        .main-header {
            font-size: 54px;
            font-weight: 800;
            text-align: center;
            margin: 8px 0 4px 0;
            letter-spacing: 0.6px;
            background: linear-gradient(90deg, #22d3ee, #60a5fa, #34d399);
            background-size: 200% 200%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: gradientShift 8s ease infinite;
        }
        .sub-header {
            font-size: 17px;
            text-align: center;
            color: var(--text-muted);
            margin-bottom: 22px;
            animation: fadeInUp 0.8s ease;
        }
        .hero-chip {
            text-align: center;
            font-size: 12px;
            letter-spacing: 2px;
            color: #7dd3fc;
            text-transform: uppercase;
            margin-top: 2px;
            animation: fadeInUp 0.7s ease;
        }
        .result-group-title {
            color: #dbeafe;
            font-weight: 700;
            font-size: 1.65rem;
            margin: 0.5rem 0 0.55rem 0;
            letter-spacing: 0.2px;
        }
        .result-card {
            background: linear-gradient(135deg, rgba(8, 47, 73, 0.40), rgba(15, 23, 42, 0.65));
            color: var(--card-text);
            padding: 16px;
            border-radius: 14px;
            margin: 10px 0;
            border: 1px solid rgba(125, 211, 252, 0.35);
            box-shadow: 0 10px 30px rgba(2, 6, 23, 0.38);
            backdrop-filter: blur(8px);
            transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
            animation: fadeInUp 0.55s ease both;
        }
        .result-card:hover {
            transform: translateY(-2px);
            border-color: rgba(52, 211, 153, 0.65);
            box-shadow: 0 15px 35px rgba(6, 182, 212, 0.18);
        }
        .result-index {
            color: #7dd3fc;
            font-weight: 700;
        }
        .result-text {
            color: var(--card-text);
            line-height: 1.45;
        }
        .gap-score-card {
            text-align: center;
            padding: 24px 20px;
            border-radius: 20px;
            margin: 24px 0;
            border: 1px solid rgba(255, 255, 255, 0.28);
            background: linear-gradient(130deg, rgba(34, 211, 238, 0.18), rgba(59, 130, 246, 0.35), rgba(52, 211, 153, 0.18));
            backdrop-filter: blur(12px);
            box-shadow: 0 18px 45px rgba(8, 47, 73, 0.45);
            animation: pulseGlow 3.2s ease-in-out infinite;
        }
        .gap-score-label {
            color: #dbeafe;
            font-size: 13px;
            letter-spacing: 2.6px;
            margin: 0;
        }
        .gap-score-value {
            color: #f8fafc;
            font-size: 52px;
            font-weight: 800;
            margin: 8px 0 0 0;
            line-height: 1;
        }
        .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {
            border-radius: 14px !important;
            border: 1px solid rgba(125, 211, 252, 0.35) !important;
            background: rgba(15, 23, 42, 0.58) !important;
            color: #e2e8f0 !important;
            transition: all 0.2s ease !important;
        }
        .stTextArea textarea:focus, .stSelectbox div[data-baseweb="select"] > div:focus-within {
            border-color: rgba(52, 211, 153, 0.8) !important;
            box-shadow: 0 0 0 1px rgba(52, 211, 153, 0.35) !important;
        }
        .main-header {
            margin-bottom: 0;
        }
        .stButton > button {
            border-radius: 14px;
            background: linear-gradient(90deg, #0ea5e9, #3b82f6, #10b981);
            background-size: 220% 220%;
            color: white;
            font-weight: 700;
            border: none;
            letter-spacing: 0.2px;
            padding: 12px 30px;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            animation: gradientShift 6s ease infinite;
            box-shadow: 0 10px 22px rgba(14, 165, 233, 0.3);
        }
        .stButton > button:hover {
            transform: translateY(-1px) scale(1.01);
            box-shadow: 0 12px 28px rgba(16, 185, 129, 0.35);
        }
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(12px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        @keyframes pulseGlow {
            0% { box-shadow: 0 14px 35px rgba(8, 47, 73, 0.45); }
            50% { box-shadow: 0 22px 50px rgba(59, 130, 246, 0.35); }
            100% { box-shadow: 0 14px 35px rgba(8, 47, 73, 0.45); }
        }
        @media (max-width: 768px) {
            .main-header { font-size: 36px; }
            .gap-score-value { font-size: 42px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="hero-chip">NEURAL DISCOVERY INTERFACE</div>', unsafe_allow_html=True)
    st.markdown('<p class="main-header">🔬 Research Gap Finder</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Identify high-potential research gaps from recent papers</p>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header(" How It Works")
        st.markdown(
            """
        **1. Enter Topic**
        - Input your research topic or research question

        **2. AI Analysis**
        - We fetch recent papers from arXiv
        - Analyze using sentence embeddings
        - Gemini/HuggingFace identifies trends and gaps

        **3. Get Results**
        - View current trends
        - Discover research gaps
        - Get project ideas and future directions
        """
        )

        st.markdown("---")
        st.header(" Features")
        st.markdown(
            """
        -  Semantic Literature Scan
        -  Gap Intelligence
        -  Project-Ready Output
        -  Exportable Reports
        -  Fast Iteration
        """
        )

        st.markdown("---")
        st.header(" API Status")
        try:
            health_response = requests.get(f"{API_URL}/health", timeout=5)
            if health_response.status_code == 200:
                st.success(" Backend Connected")
            else:
                st.warning(" Backend Issue")
        except Exception:
            st.error(" Backend Offline")
            st.markdown(
                """
            **Make sure the backend is running:**

```bash
cd backend
python app.py
```

            **Or set custom API URL:**

```bash
export STREAMLIT_API_URL=http://your-api:5000
```
            """
            )

        st.markdown("---")
        st.header(" Search History")
        history = st.session_state.search_history
        if history:
            st.download_button(
                label="Download History (.txt)",
                data=format_history_as_text(history),
                file_name="research_gap_history.txt",
                mime="text/plain",
                key="download_history_btn",
                use_container_width=True,
            )

            if st.button("Clear History", key="clear_history_btn", use_container_width=True):
                st.session_state.search_history = []
                st.session_state.latest_results = None
                st.success("History cleared")
                st.rerun()

            for i, entry in enumerate(history, 1):
                topic_text = entry.get("topic", "Untitled search")
                short_topic = topic_text if len(topic_text) <= 45 else topic_text[:45] + "..."
                with st.expander(f"{i}. {short_topic}", expanded=False):
                    st.caption(entry.get("timestamp", ""))
                    st.write(f"Language: {entry.get('language', 'N/A')}")
                    st.write(f"Gap Score: {entry.get('gap_score', 'N/A')}")
        else:
            st.caption("No search history yet.")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        language = st.selectbox(
            "Select output language:",
            [
                "English",
                "Hindi",
                "Spanish",
                "French",
                "German",
                "Portuguese",
                "Arabic",
                "Chinese",
                "Japanese",
                "Korean",
            ],
            index=0,
        )
        topic = st.text_area(
            "Enter your research topic:",
            placeholder="Example: Multimodal foundation models for low-resource healthcare diagnostics",
            height=100,
        )

    if st.button(" Analyze Topic", use_container_width=True):
        if not topic.strip():
            st.error("Please enter a research topic!")
        else:
            with st.spinner("Fetching papers from arXiv... This may take a minute."):
                try:
                    results = analyze_topic(topic, language)
                    save_search_to_history(topic, language, results)
                    st.rerun()
                except requests.exceptions.ConnectionError:
                    st.error(" Cannot connect to backend. Please make sure the backend is running!")
                    st.info(" Run: cd backend && python app.py")
                except requests.exceptions.Timeout:
                    st.error("Request timed out. Please try again.")
                except Exception as e:
                    st.error(f" Error: {str(e)}")

    if st.session_state.latest_results:
        st.success("Analysis Complete!")
        display_analysis_results(st.session_state.latest_results)

    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #888; padding: 20px;">
            <p>Research Gap Finder | Powered by arXiv + Gemini/HuggingFace + Sentence Transformers</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
