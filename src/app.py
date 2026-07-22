"""
app.py
------
Streamlit chat interface for the RAG-based Socratic Tutor.

Run with:
    streamlit run src/app.py
"""

import os
import sys
import streamlit as st
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.socratic_tutor import SocraticTutor, TutorSession
from src.retriever import Retriever

load_dotenv()

st.set_page_config(page_title="Socratic RAG Tutor", page_icon="🎓", layout="centered")

st.title("🎓 Socratic AI Teaching Assistant")
st.caption("Ask a question about the course material. I won't just give you the answer - I'll help you find it.")

# --- Setup / sanity checks -------------------------------------------------

if "ANTHROPIC_API_KEY" not in os.environ or not os.environ["ANTHROPIC_API_KEY"]:
    st.error(
        "No ANTHROPIC_API_KEY found. Copy `.env.example` to `.env` and add your key, "
        "then restart the app."
    )
    st.stop()

retriever_check = Retriever()
if not retriever_check.is_ready():
    st.warning(
        "No indexed course material found yet. Run `python src/ingest.py` first "
        "to build the index from files in the `data/` folder, then refresh this page."
    )
    st.stop()

# --- Session state -----------------------------------------------------------

if "tutor" not in st.session_state:
    st.session_state.tutor = SocraticTutor()

if "session" not in st.session_state:
    st.session_state.session = TutorSession()

if "display_history" not in st.session_state:
    st.session_state.display_history = []

session: TutorSession = st.session_state.session

# --- Sidebar: progress + reset ----------------------------------------------

with st.sidebar:
    st.subheader("Progress")
    st.metric("Hints used", session.hint_count)
    st.progress(min(session.hint_count / 4, 1.0))
    if session.resolved:
        st.success("Concept resolved! Ask a new question anytime.")
    if st.button("🔄 Start a new question"):
        st.session_state.session = TutorSession()
        st.session_state.display_history = []
        st.rerun()

# --- Chat history ------------------------------------------------------------

for turn in st.session_state.display_history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("sources"):
            with st.expander("📖 Sources used"):
                for s in turn["sources"]:
                    st.caption(s)

# --- Chat input ----------------------------------------------------------------

if prompt := st.chat_input("Ask about the course material..."):
    st.session_state.display_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking of a good guiding question..."):
            result = st.session_state.tutor.ask(session, prompt)
        st.markdown(result["reply"])
        if result["sources"]:
            with st.expander("📖 Sources used"):
                for s in result["sources"]:
                    st.caption(s)

    st.session_state.display_history.append({
        "role": "assistant",
        "content": result["reply"],
        "sources": result["sources"],
    })
    st.rerun()
