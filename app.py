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
OUTPUT_FILENAME = "edited_document" # Base name for download

# --- LLM API Configuration ---
API_KEY = os.environ.get("GEMINI_API_KEY")
API_URL_TEXT = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={API_KEY}"
API_URL_IMAGEN = f"https://generativelanguage.googleapis.com/v1beta/models/imagen-3.0-generate-002:predict?key={API_KEY}"

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
            
            # Extract text/JSON result from Gemini 2.5 response
            if 'candidates' in result and result['candidates']:
                return result['candidates'][0]['content']['parts'][0]['text']
            
            # Extract image result from Imagen 3.0 response
            if 'predictions' in result and result['predictions']:
                return result['predictions'][0]['bytesBase64Encoded']
            
            return None

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

def generate_edited_image(image_buffer, edited_text):
    """Uses Imagen to replace text on the original image."""
    st.info("🎨 Sending original image and edited text to AI for re-rendering...")
    
    base64_image = base64.b64encode(image_buffer).decode('utf-8')

    prompt = (
        f"Based on the input image, replace the existing text with the following new text. "
        f"Maintain the original image's style, fonts, colors, and layout perfectly. "
        f"New Content:\n---\n{edited_text}"
    )

    payload = {
        "instances": {
            "prompt": prompt,
            "image_bytes_base64": base64_image
        },
        "parameters": {"sampleCount": 1}
    }

    # Use call_gemini_api which handles the Imagen endpoint
    with st.spinner("AI is generating the edited image (this may take up to 30 seconds)..."):
        return call_gemini_api(API_URL_IMAGEN, payload)

# --- STREAMLIT UI ---

def main_app():
    st.set_page_config(layout="wide", page_title="Document AI Co-Editor")

    # --- Initialize Session State (Fixes KeyError on initial load) ---
    if 'editing_started' not in st.session_state: st.session_state['editing_started'] = False
    if 'document_type' not in st.session_state: st.session_state['document_type'] = 'None'
    # The crucial fix for the image buffer key error:
    if 'original_image_buffer' not in st.session_state: st.session_state['original_image_buffer'] = None
    if 'editor_text' not in st.session_state: st.session_state['editor_text'] = ""
    # ------------------------------------------------------------------

    st.title("📄 AI Image & Document Editor (Re-render Text)")
    st.markdown("Upload an image or document, and the AI will extract, edit, and re-render the content based on your transcript.")

    if not API_KEY:
        st.error("⚠️ **Deployment Error:** The `GEMINI_API_KEY` environment variable is not set. Please configure it.")

    # 1. Inputs: File and Transcript
    st.subheader("1. Upload Files and Transcript")
    
    col_file, col_transcript = st.columns([1, 1])
    
    with col_file:
        uploaded_file = st.file_uploader(
            "Upload Target Document (PDF, PNG, or JPG)",
            type=["pdf", "png", "jpg", "jpeg"],
            key='uploaded_file'
        )
    
    with col_transcript:
        transcript = st.text_area(
            "Paste Meeting Transcript Here",
            height=200,
            key='transcript_area'
        )

    st.markdown("---")
    
    # 2. Execution Button (Start Analysis)
    if st.button("Start Analysis and Editing 🚀", disabled=(not uploaded_file or not transcript or not API_KEY)):
        
        file_buffer = uploaded_file.getvalue()
        file_type = uploaded_file.type

        # Extraction Logic based on file type
        if file_type == 'application/pdf':
            extracted_text = extract_text_from_pdf(io.BytesIO(file_buffer))
            st.session_state['document_type'] = 'PDF'
            # PDF doesn't have an original image buffer for re-rendering
            st.session_state['original_image_buffer'] = None 
        elif file_type in ['image/png', 'image/jpeg']:
            extracted_text = extract_text_from_image(file_buffer, file_type)
            st.session_state['document_type'] = 'Image'
            st.session_state['original_image_buffer'] = file_buffer # Store buffer for re-render
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

    # --- Editing and Re-rendering Interface ---
    if st.session_state.get('editing_started'):
        st.subheader("2. Text Editing & Visual Re-render")

        col_editor, col_visual_output = st.columns([1, 1])
        document_type = st.session_state.get('document_type', 'File')
        
        # A. Text Editor (User editable area)
        with col_editor:
            st.info(f"Source Document Type: {document_type}")
            if 'ai_suggestion' in st.session_state:
                st.success("✅ AI has generated suggested edits.")
            
            # Allow user to edit the text
            # We use the key 'final_editor' to keep the text persistent
            final_edited_text = st.text_area(
                "Final Edited Text (Review/Modify below):",
                value=st.session_state['editor_text'],
                height=450,
                key='final_editor'
            )
            
            # Update session state text whenever the user types/the editor value changes
            st.session_state['editor_text'] = final_edited_text

        # B. Output Visuals & Re-render Button
        with col_visual_output:
            st.warning("Preview/Re-render Area")

            if document_type == 'Image' and st.session_state['original_image_buffer']:
                # Show the original image as a reference
                st.image(st.session_state['original_image_buffer'], caption="Original Image Reference", use_container_width=True)
                
                if st.button("🖼️ Generate Edited Image Output", key='generate_image_button'):
                    # The image generation function uses the text currently in the editor
                    image_base64_data = generate_edited_image(
                        st.session_state['original_image_buffer'], 
                        final_edited_text
                    )
                    
                    if image_base64_data:
                        # Display the new image and allow download
                        image_url = f"data:image/jpeg;base64,{image_base64_data}"
                        st.image(image_url, caption="AI Re-rendered Image Output", use_container_width=True)
                        st.download_button(
                            label="Download Re-rendered Image",
                            data=base64.b64decode(image_base64_data),
                            file_name="edited_image_output.jpeg",
                            mime="image/jpeg"
                        )
            
            elif document_type == 'PDF':
                st.info("PDF: Output is text only. No image re-rendering is possible.")
            
            else:
                st.warning("No visual output available.")

        st.markdown("---")

        # 3. Final Text Output (Always available)
        st.subheader("3. Final Edited Text Output (For Manual Use)")
        
        st.download_button(
            label="Download Final Edited Text (.txt)",
            data=st.session_state['editor_text'],
            file_name=f"{OUTPUT_FILENAME}.txt",
            mime="text/plain"
        )

if __name__ == "__main__":
    main_app()
