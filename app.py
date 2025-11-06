import streamlit as st
import os
import json
import time
import requests
import io
import sys

# New library for PDF reading
try:
    from pypdf import PdfReader
except ImportError:
    st.error("Missing 'pypdf' library. Please add it to requirements.txt.")
    sys.exit()

# --- Configuration ---
OUTPUT_FILENAME = "edited_text_output.txt"
# This file name is for the initial PDF upload only
PDF_INPUT_FILENAME = "input_document.pdf" 

# --- LLM API Configuration ---
API_KEY = os.environ.get("GEMINI_API_KEY")
API_URL_TEXT = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={API_KEY}"

# --- CORE LOGIC FUNCTIONS ---

def call_gemini_api(url, payload):
    """Helper function to call the Gemini API with retry logic."""
    max_retries = 3
    delay = 2 
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
            response.raise_for_status()
            result = response.json()
            if 'candidates' not in result or not result['candidates']: return None
            text = result['candidates'][0]['content']['parts'][0]['text']
            return text
        except requests.exceptions.HTTPError as e:
            st.error(f"API Error (Attempt {attempt + 1}): {e.response.status_code}")
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
            else:
                st.error("Max retries reached. Aborting AI call.")
                raise
        except Exception as e:
            st.error(f"An unexpected error occurred during API call: {e}")
            raise
    return None

def extract_text_from_pdf(pdf_buffer):
    """Reads PDF buffer and returns all text, separated by page markers."""
    reader = PdfReader(pdf_buffer)
    all_text = []
    for i, page in enumerate(reader.pages):
        all_text.append(f"--- PAGE {i + 1} START ---")
        all_text.append(page.extract_text() or " [No Text Extracted on this page] ")
        all_text.append(f"--- PAGE {i + 1} END ---")
        all_text.append("")
    return "\n".join(all_text)

def generate_ai_suggestions(transcript, extracted_text):
    """Calls Gemini to rewrite the extracted PDF text based on the transcript."""
    st.info("🧠 AI is analyzing the PDF text and transcript for suggested edits...")

    MAX_TRANSCRIPT_LENGTH = 8000
    if len(transcript) > MAX_TRANSCRIPT_LENGTH:
        head = transcript[:4000]
        tail = transcript[-4000:]
        transcript_to_send = f"{head}\n\n[... TRANSCRIPT TRUNCATED ...]\n\n{tail}"
    else:
        transcript_to_send = transcript

    system_prompt = (
        "You are a document editor. Your task is to rewrite the 'ORIGINAL PDF TEXT' "
        "to reflect the key decisions and content from the 'MEETING TRANSCRIPT'. "
        "Maintain the original structure and page markers (--- PAGE X START ---). "
        "Only update sections that are relevant to the transcript. Provide ONLY the final rewritten text."
    )

    user_query = (
        f"MEETING TRANSCRIPT:\n---\n{transcript_to_send}\n---\n\n"
        f"ORIGINAL PDF TEXT (Maintain Structure/Page Markers):\n---\n{extracted_text}\n---\n"
        "Provide the final, fully edited document text:"
    )

    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]}
    }

    with st.spinner("Rewriting document content..."):
        return call_gemini_api(API_URL_TEXT, payload)


# --- STREAMLIT UI ---

def main_app():
    st.set_page_config(layout="wide", page_title="PDF AI Co-Editor")

    st.title("📄 PDF AI Co-Editor (Visual Reference Strategy)")
    st.markdown("This tool extracts text from your PDF, lets you edit it with AI help, and provides the final edited text.")

    if not API_KEY:
        st.error("⚠️ **Deployment Error:** The `GEMINI_API_KEY` environment variable is not set.")

    # 1. Inputs: PDF, Visual Reference Image, and Transcript
    st.subheader("1. Upload Files and Transcript")
    
    col_file, col_visual = st.columns(2)
    
    with col_file:
        uploaded_pdf = st.file_uploader(
            "Upload Target PDF Document",
            type="pdf",
            key='uploaded_pdf'
        )
    
    with col_visual:
        uploaded_image = st.file_uploader(
            "Upload Visual Reference (PNG/JPEG image of the page/slide)",
            type=["png", "jpg", "jpeg"],
            key='uploaded_image',
            help="Upload a screenshot of the page for live visual context while editing."
        )

    transcript = st.text_area(
        "Paste Meeting Transcript Here",
        height=200,
        key='transcript_area'
    )

    st.markdown("---")
    
    # 2. Execution Button
    if st.button("Start Analysis and Editing 🚀", disabled=(not uploaded_pdf or not transcript or not API_KEY)):
        if uploaded_pdf:
            st.session_state['pdf_buffer'] = uploaded_pdf.getvalue()
            
            with st.spinner("Extracting text from PDF..."):
                extracted_text = extract_text_from_pdf(io.BytesIO(st.session_state['pdf_buffer']))
                st.session_state['extracted_text'] = extracted_text
                
            st.session_state['editing_started'] = True
            st.session_state['ai_suggestion'] = generate_ai_suggestions(transcript, extracted_text)
            
            # Use AI suggestion as the initial text for the editor if available
            st.session_state['editor_text'] = st.session_state.get('ai_suggestion', extracted_text)
            
            st.rerun()

    # --- Editing Interface (Runs after analysis is complete) ---
    if st.session_state.get('editing_started'):
        st.subheader("2. Live Visual Reference & Text Editing")

        col_visual_ref, col_editor = st.columns(2)

        # A. Visual Reference (The "Live Preview" replacement)
        with col_visual_ref:
            if uploaded_image:
                st.image(uploaded_image, caption="Visual Reference (Live Preview)", use_column_width=True)
            else:
                st.warning("Upload a visual image to get a live reference while editing.")
                st.info("The text is organized by page markers (--- PAGE X START ---).")

        # B. Text Editor
        with col_editor:
            
            # Display AI suggestion status
            if 'ai_suggestion' in st.session_state:
                st.success("✅ AI has analyzed the transcript and generated suggested edits.")
            else:
                st.info("Editing original extracted text.")
                
            # Allow user to edit the text
            final_edited_text = st.text_area(
                "Final Edited Document Text (Review/Modify below):",
                value=st.session_state['editor_text'],
                height=600,
                key='final_editor'
            )

        st.markdown("---")

        # 3. Final Output Button
        st.subheader("3. Final Output")
        
        # Download button provides the final text output
        st.download_button(
            label="Download Final Edited Text (.txt)",
            data=final_edited_text,
            file_name=OUTPUT_FILENAME,
            mime="text/plain"
        )
        st.info("The output is a plain text file. Use this text to manually recreate your final PDF document.")

if __name__ == "__main__":
    # Ensure 'requests' is available for the API call
    if 'requests' not in sys.modules:
        try:
            import requests
        except ImportError:
            # This is handled during initial deployment with requirements.txt
            pass

    main_app()
