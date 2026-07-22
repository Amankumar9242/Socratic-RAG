# 🎓 Socratic RAG Teaching Assistant

A working RAG-based AI teaching assistant that doesn't just answer questions —
it **guides students toward the answer themselves**, the way a good human tutor
would. Built with Claude, Chroma, and Streamlit.

## How it works

1. **Ingestion** — your course material (`.txt` / `.pdf`) is chunked by section
   and embedded into a local Chroma vector database.
2. **Retrieval** — when a student asks a question, the most relevant chunks are
   retrieved.
3. **Socratic engine** — instead of answering directly, Claude is prompted (with
   the retrieved context as ground truth) to ask ONE guiding question per turn.
   A hint counter tracks progress, and a "leak checker" pass catches cases where
   the model tries to give away the answer anyway.
4. **Reveal** — after a configurable number of hints (default 4), the tutor
   explains the answer directly, citing the source material.

## Project structure

```
socratic-tutor/
├── data/                  # Drop your course material here (.txt / .pdf)
│   └── sample_notes.txt   # Sample calculus notes to test with out of the box
├── chroma_db/             # Local vector DB (created automatically)
├── src/
│   ├── ingest.py          # Chunking + embedding + indexing
│   ├── retriever.py       # Query wrapper around Chroma
│   ├── socratic_tutor.py  # Core Socratic prompting + hint-state engine
│   └── app.py             # Streamlit chat UI
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

**1. Install dependencies** (Python 3.10+ recommended):

```bash
cd socratic-tutor
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**2. Add your Anthropic API key:**

```bash
cp .env.example .env
```

Then edit `.env` and paste your key:
```
ANTHROPIC_API_KEY=sk-ant-...
```

**3. Add course material.**
A sample file (`data/sample_notes.txt`, an intro-to-derivatives lesson) is
already included so you can test immediately. To use your own material, drop
`.txt` or `.pdf` files into `data/`.

**4. Build the index:**

```bash
python src/ingest.py
```

You should see something like `Indexed 6 chunks from 1 document(s).`
Re-run this any time you add or change files in `data/`.

**5. Launch the app:**

```bash
streamlit run src/app.py
```

This opens the chat interface in your browser (usually `http://localhost:8501`).

## Try it out

With the included sample notes, try asking:
- "What's the derivative of x^3 + 5x?"
- "How do I differentiate (3x+1)^2?"
- "Why can't I just multiply the derivatives in a product?"

Notice the tutor won't just hand you the formula — it'll ask a question back
first, referencing the actual retrieved material.

## Configuration

Edit `.env` to change:
- `CLAUDE_MODEL` — defaults to `claude-sonnet-4-6`
- `MAX_HINTS_BEFORE_REVEAL` — how many guiding questions before the tutor
  explains the answer outright (default `4`)

## Extending this project

Some natural next steps if you want to go further:
- **Multi-subject support** — store a `subject` field in metadata and filter
  retrieval by course.
- **Persistent student profiles** — track which concepts a student struggles
  with across sessions, not just within one.
- **Difficulty levels** — let students choose "gentle hints" vs. "strict
  Socratic mode."
- **Voice mode** — add Whisper for speech-to-text and TTS for spoken tutoring.
- **Eval harness** — log hint counts and resolution rates to measure whether
  the Socratic mode actually improves learning outcomes vs. direct answers.

## Notes on cost/privacy

- Embeddings run **locally** (`sentence-transformers`, free, no API calls).
- Only the generation step (guiding questions + leak-check) calls the
  Anthropic API, so each student turn costs roughly 2 small API calls.
