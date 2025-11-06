import streamlit as st
import os
import json
import time
import requests
import io
import sys
import base64

# New library for PDF reading
try:
    from pypdf import PdfReader
except ImportError:
    # This dependency must be in requirements.txt for deployment
    st.error("Missing 'pypdf' library. Please ensure requirements.txt is up-to-date.")
    sys.exit()

# --- Configuration ---
OUTPUT_FILENAME = "edited_text_output.txt"

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

def extract_text_from_image(image_buffer, mime_type):
    """Uses Gemini Vision to extract text from an image."""
    st.info("🧠 AI Vision is reading text from the image...")
    
    base64_image = base64.b64encode(image_buffer).decode('utf-8')
    
    user_query = "Extract all text present in this image, retaining all original formatting like line breaks and bullet points. Do not add any commentary, only the raw text content."
    
    payload = {
        "contents": [{
            "parts": [
                {"text": user_query},
                {"inlineData": {"mimeType": mime_type, "data": base64_image}}
            ]
        }]
    }
    
    # We use the text endpoint, but the payload includes image data for vision
    extracted_text = call_gemini_api(API_URL_TEXT, payload)
    
    if extracted_text:
        return f"--- IMAGE TEXT START ---\n{extracted_text}\n--- IMAGE TEXT END ---"
    return " [Failed to extract text from image.] "

def generate_ai_suggestions(transcript, extracted_text):
    """Calls Gemini to rewrite the extracted text based on the transcript."""
    st.info("🧠 AI is analyzing the transcript and your document content for suggested edits...")

    MAX_TRANSCRIPT_LENGTH = 8000
    if len(transcript) > MAX_TRANSCRIPT_LENGTH:
        head = transcript[:4000]
        tail = transcript[-4000:]
        transcript_to_send = f"{head}\n\n[... TRANSCRIPT TRUNCATED ...]\n\n{tail}"
    else:
        transcript_to_send = transcript

    system_prompt = (
        "You are a document editor. Your task is to rewrite the 'ORIGINAL DOCUMENT TEXT' "
        "to reflect the key decisions and content from the 'MEETING TRANSCRIPT'. "
        "Maintain the original structure (page markers/line breaks). "
        "Only update sections that are relevant to the transcript. Provide ONLY the final rewritten text."
    )

    user_query = (
        f"MEETING TRANSCRIPT:\n---\n{transcript_to_send}\n---\n\n"
        f"ORIGINAL DOCUMENT TEXT (Maintain Structure):\n---\n{extracted_text}\n---\n"
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
    st.set_page_config(layout="wide", page_title="Document AI Co-Editor")

    st.title("📄 Document AI Co-Editor (Image/PDF)")
    st.markdown("Upload your document/image, and the AI will extract and rewrite the text based on your transcript.")

    if not API_KEY:
        st.error("⚠️ **Deployment Error:** The `GEMINI_API_KEY` environment variable is not set. Please configure it.")

    # 1. Inputs: File, Visual Reference Image, and Transcript
    st.subheader("1. Upload Files and Transcript")
    
    col_file, col_visual = st.columns(2)
    
    with col_file:
        uploaded_file = st.file_uploader(
            "Upload Target Document (PDF, PNG, or JPG)",
            type=["pdf", "png", "jpg", "jpeg"],
            key='uploaded_file'
        )
    
    with col_visual:
        # User can upload the same file as the reference, or a higher quality screenshot
        uploaded_image = st.file_uploader(
            "Upload Visual Reference (For Live Preview)",
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
    if st.button("Start Analysis and Editing 🚀", disabled=(not uploaded_file or not transcript or not API_KEY)):
        
        file_buffer = uploaded_file.getvalue()
        file_type = uploaded_file.type

        # Extraction Logic based on file type
        if file_type == 'application/pdf':
            extracted_text = extract_text_from_pdf(io.BytesIO(file_buffer))
            st.session_state['document_type'] = 'PDF'
        elif file_type in ['image/png', 'image/jpeg']:
            extracted_text = extract_text_from_image(file_buffer, file_type)
            st.session_state['document_type'] = 'Image'
        else:
            st.error("Unsupported file type.")
            return

        st.session_state['extracted_text'] = extracted_text
        st.session_state['editing_started'] = True
        
        # Generate AI suggestions based on extracted text and transcript
        st.session_state['ai_suggestion'] = generate_ai_suggestions(transcript, extracted_text)
        st.session_state['editor_text'] = st.session_state.get('ai_suggestion', extracted_text)
        
        st.success("Analysis complete. Scroll down to edit.")
        st.rerun()

    # --- Editing Interface (Runs after analysis is complete) ---
    if st.session_state.get('editing_started'):
        st.subheader("2. Live Visual Reference & Text Editing")

        col_visual_ref, col_editor = st.columns(2)
        document_type = st.session_state.get('document_type', 'File')
        
        # A. Visual Reference (The "Live Preview" replacement)
        with col_visual_ref:
            st.info(f"Source Document Type: {document_type}")
            if uploaded_image:
                st.image(uploaded_image, caption="Visual Reference (Live Preview)", use_column_width=True)
            elif uploaded_file and document_type == 'Image':
                st.image(uploaded_file, caption="Visual Reference (Uploaded Image)", use_column_width=True)
            else:
                st.warning("Upload an image for a visual reference.")

        # B. Text Editor
        with col_editor:
            
            if 'ai_suggestion' in st.session_state:
                st.success("✅ AI has analyzed the transcript and generated suggested edits.")
            
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
        
        st.warning("⚠️ **Important:** The application cannot save the text back into the image. You must use the text file below to manually update your original image/document.")
        
        # Download button provides the final text output
        st.download_button(
            label="Download Final Edited Text (.txt)",
            data=final_edited_text,
            file_name=OUTPUT_FILENAME,
            mime="text/plain"
        )

if __name__ == "__main__":
    # Ensure 'requests' is available for the API call
    if 'requests' not in sys.modules:
        try:
            import requests
        except ImportError:
            pass

    main_app()
