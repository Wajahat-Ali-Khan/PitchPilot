import os
import uuid
import json
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st

from agent.agent_chain import PitchAgent
from db.database import Database
from pdf.deck_generator import DeckGenerator

load_dotenv()

# Config
TEST_INPUT = "Airbnb for pets"

st.set_page_config(page_title="PitchPilot — MVP", layout="centered")
st.title("PitchPilot — MVP")

idea_text = st.text_area("Enter your startup idea (one-liner or short paragraph):", value=TEST_INPUT, height=120)
generate_button = st.button("Generate Pitch Deck")

db = Database(db_path=os.getenv("PITCHPILOT_DB", "pitchpilot.db"))
agent = PitchAgent()
deck_gen = DeckGenerator(template_path=os.path.join("pdf", "templates", "deck_template.html"))

if generate_button:
    if not idea_text.strip():
        st.error("Idea text cannot be empty.")
    else:
        with st.spinner("Running AI agent and generating slides..."):
            try:
                slides = agent.run_workflow(idea_text)
            except Exception as e:
                st.error(f"Agent failed: {e}")
                raise

        # Validate slides structure
        if not isinstance(slides, list):
            st.error("Agent did not return a slide list. Check logs.")
        else:
            # Save to DB
            deck_id = str(uuid.uuid4())
            pdf_filename = f"pitch_{deck_id}.pdf"
            pdf_output_path = os.path.join("pdf", "output")
            os.makedirs(pdf_output_path, exist_ok=True)
            pdf_path = os.path.join(pdf_output_path, pdf_filename)

            try:
                # Generate PDF
                deck_gen.generate_pdf(slides, title=idea_text, output_path=pdf_path)

                # Save record
                db.save_deck(
                    deck_id=deck_id,
                    idea_text=idea_text,
                    slides_json=json.dumps(slides, ensure_ascii=False),
                    pdf_path=pdf_path,
                    created_at=datetime.utcnow(),
                )
            except Exception as e:
                st.error(f"Failed to generate/save PDF: {e}")
                raise

            # Offer download
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()

            st.success("Pitch deck generated.")
            st.download_button(
                label="Download Pitch Deck (PDF)",
                data=pdf_bytes,
                file_name=pdf_filename,
                mime="application/pdf",
            )

# Small helper to show last 3 decks
if st.checkbox("Show recent pitch decks (DB)"):
    rows = db.list_recent(3)
    for r in rows:
        st.markdown(f"**{r['id']}** — {r['idea_text']} — {r['created_at']}")
