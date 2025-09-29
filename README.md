# PitchPilot — MVP

Purpose: Generate investor-ready pitch decks automatically using an agentic LangChain workflow backed by HuggingFace models.

## Features (MVP)
- Input: single-line or short paragraph idea.
- Agentic workflow (LangChain + HuggingFace):
  1. Extract category/problem/solution
  2. Market research & trends
  3. Business model suggestions
  4. Generate 8–10 slide contents (JSON)
- Output: Downloadable PDF created from HTML slides via Jinja2 + WeasyPrint
- Frontend: Streamlit app with single input and PDF download
- Database: SQLite storing idea, slides JSON, pdf path, timestamp

## Quickstart (local)
1. Clone the project files into a folder `pitchpilot/`.
2. Create a Python venv and activate it.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
4. Create .env from .env.example. Add HF_API_KEY for Hugging Face Inference API if available. If not, ensure LOCAL_MODEL is set and a compatible model is available locally (e.g., gpt2).

5. Run Streamlit:
   ```bash
   streamlit run app.py


6. Use the test input "Airbnb for pets" (pre-filled) or enter your idea and click "Generate Pitch Deck".

## Files

- app.py — Streamlit frontend.

- agent/agent_chain.py — LangChain orchestration and LLM wrapper for HuggingFace Inference API.

- agent/prompts.py — Prompt templates for each step.

- db/database.py — SQLite helper to persist pitch decks.

- pdf/deck_generator.py — Jinja2 + WeasyPrint PDF generator.

- pdf/templates/deck_template.html — HTML slide template.

- .env.example — Example env file.