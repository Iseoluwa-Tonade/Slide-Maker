import streamlit as st
import os
import json
import time
import requests
import io
from pptx import Presentation
from pptx.enum.dml import MSO_COLOR_TYPE
from pptx.enum.text import PP_ALIGN

# --- Configuration ---
# Set the default name for the output file
OUTPUT_FILENAME = "edited_presentation.pptx"

# --- LLM API Configuration (Read from Streamlit Secrets/Environment) ---
# NOTE: The value of this key will be read securely from Streamlit's environment settings.
API_KEY = os.environ.get("GEMINI_API_KEY")

# Model and API endpoint for structured JSON output
API_URL_JSON = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={API_KEY}"
# A separate endpoint/config for simple text generation
API_URL_TEXT = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={API_KEY}"

# --- CORE LOGIC FUNCTIONS (Adapted for Streamlit) ---

def apply_text_formatting(target_run, source_font):
    """Reapplies font size, name, bold, and color from source_font to target_run."""
    if source_font:
        # 1. Size and Name
        if source_font.size: target_run.font.size = source_font.size
        if source_font.name: target_run.font.name = source_font.name
        target_run.font.bold = source_font.bold

        # 2. Color Preservation Logic
        color_type = source_font.color.type
        if color_type == MSO_COLOR_TYPE.RGB:
            target_run.font.color.rgb = source_font.color.rgb
        elif color_type == MSO_COLOR_TYPE.SCHEME:
            target_run.font.color.theme_color = source_font.color.theme_color
            if hasattr(source_font.color, 'brightness') and source_font.color.brightness is not None:
                target_run.font.color.brightness = source_font.color.brightness

def replace_text_safely(shape, new_text):
    """Replaces text in a shape while preserving the formatting of the first run."""
    text_frame = shape.text_frame

    if not text_frame.paragraphs:
        p0 = text_frame.add_paragraph()
    else:
        p0 = text_frame.paragraphs[0]

    r0 = p0.runs[0].font if p0.runs else None

    original_alignment = p0.alignment
    original_level = p0.level

    while len(text_frame.paragraphs) > 1:
        p = text_frame.paragraphs[-1]
        if hasattr(p, '_element'):
            p._element.getparent().remove(p._element)
        else:
            break
    p0.clear()

    new_lines = new_text.split('\n')

    if new_lines:
        run = p0.add_run()
        run.text = new_lines[0]
        p0.alignment = original_alignment
        p0.level = original_level
        apply_text_formatting(run, r0)

        for line in new_lines[1:]:
            p = text_frame.add_paragraph()
            p.text = line
            p.level = original_level
            p.alignment = original_alignment

            if p.runs:
                apply_text_formatting(p.runs[0], r0)

def extract_presentation_text_blocks(prs):
    """
    Scans the presentation and returns a numbered list of all non-empty text blocks
    and a mapping to their original shape objects.
    """
    presentation_blocks = []
    shape_map = []

    for slide_idx, slide in enumerate(prs.slides):
        for shape in slide.shapes:
            if shape.has_text_frame:
                current_text = "\n".join(p.text for p in shape.text_frame.paragraphs if p.text.strip())

                if current_text.strip():
                    presentation_blocks.append({
                        "id": len(presentation_blocks),
                        "slide_number": slide_idx + 1,
                        "current_text": current_text,
                        "shape_name": shape.name,
                    })
                    shape_map.append(shape)

    return presentation_blocks, shape_map

def call_gemini_api(url, payload):
    """Helper function to call the Gemini API with retry logic."""
    max_retries = 3
    delay = 2 # seconds

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
            response.raise_for_status()

            result = response.json()

            if 'candidates' not in result or not result['candidates']:
                 return None

            json_text = result['candidates'][0]['content']['parts'][0]['text']

            if not json_text:
                 return None

            return json_text

        except requests.exceptions.HTTPError as e:
            # Handle 400 errors (like token limits) specifically
            st.error(f"API Error (Attempt {attempt + 1}): {e.response.status_code}")
            if attempt < max_retries - 1:
                st.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2
            else:
                st.error("Max retries reached. Aborting AI call.")
                raise
        except Exception as e:
            st.error(f"An unexpected error occurred during API call: {e}")
            raise
    return None

@st.cache_data(show_spinner=False)
def generate_ai_mapping(_transcript, presentation_blocks, api_key):
    """
    Calls the Gemini API to analyze the transcript and map the extracted data
    to the most appropriate existing text blocks. Caches result based on transcript hash.
    """
    if not api_key:
        st.error("Gemini API Key is not configured.")
        return []

    # Use the API URL defined globally
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={api_key}"

    # --- Truncate long transcripts to avoid token limits ---
    MAX_TRANSCRIPT_LENGTH = 8000
    transcript_to_send = _transcript
    if len(_transcript) > MAX_TRANSCRIPT_LENGTH:
        head = _transcript[:4000]
        tail = _transcript[-4000:]
        transcript_to_send = f"{head}\n\n[... TRANSCRIPT TRUNCATED ...]\n\n{tail}"

    schema = {
        "type": "ARRAY",
        "description": "A list of text replacements. Only include blocks that require an update based on the transcript.",
        "items": {
            "type": "OBJECT",
            "properties": {
                "original_index": {"type": "NUMBER", "description": "The 'id' number of the text block to replace from the PRESENTATION TEXT BLOCKS list."},
                "new_text": {"type": "STRING", "description": "The concise, new text or bulleted content derived from the transcript. Use '\\n' for new lines/bullets."}
            },
            "required": ["original_index", "new_text"]
        }
    }

    system_prompt = (
        "You are a Presentation Automation Specialist. Analyze the TRANSCRIPT and the PRESENTATION TEXT BLOCKS. "
        "Your goal is to replace the generic or outdated text in the presentation with key concepts, data, and next steps "
        "from the transcript. You must use the provided JSON schema. "
        "Only create entries for text blocks that need modification. Do NOT create entries for blocks that should remain unchanged. "
        "Ensure the 'new_text' is ready for direct insertion into a PowerPoint slide."
    )

    presentation_list_text = json.dumps(presentation_blocks, indent=2)
    user_query = (
        f"MEETING TRANSCRIPT:\n---\n{transcript_to_send}\n---\n\n"
        f"PRESENTATION TEXT BLOCKS (Index to Map to):\n---\n{presentation_list_text}\n---\n"
        "Generate a JSON array of objects mapping the 'original_index' to the derived 'new_text'."
    )

    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": schema
        }
    }

    with st.spinner("Analyzing transcript and mapping changes with Gemini..."):
        json_text = call_gemini_api(url, payload)

    if json_text:
        return json.loads(json_text)
    return []

def process_batch_edits(prs, edited_block_lines, presentation_blocks, shape_map):
    """Parses the bulk edited text block and applies changes."""
    applied_count = 0

    for line in edited_block_lines:
        try:
            match = line.split('|', 3)
            if len(match) < 4:
                continue

            id_str = match[0].strip()[1:]
            block_id = int(id_str.split('|')[0])

            new_text_raw = match[3].strip()
            # Unescape newlines
            new_text = new_text_raw.replace(' \\n ', '\n').replace('\\n', '\n')

            if 0 <= block_id < len(shape_map):
                shape = shape_map[block_id]

                if not new_text.strip():
                    new_text = ""

                replace_text_safely(shape, new_text)
                applied_count += 1

        except Exception as e:
            st.warning(f"Could not parse or apply change for line: {line}. Skipping. Error: {e}")

    return applied_count

# --- STREAMLIT UI ---

def main_app():
    st.set_page_config(layout="wide", page_title="PPTX AI Co-Editor")

    st.title("📄 AI PowerPoint Automation Tool")
    st.markdown("Upload your PowerPoint template and paste a meeting transcript to automatically update the text.")

    if not API_KEY:
        st.error("⚠️ **Deployment Error:** The `GEMINI_API_KEY` environment variable is not set. Please configure it for the AI modes to function.")

    # 1. File Uploader and Transcript Input
    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload PowerPoint File (.pptx)",
            type="pptx",
            help="Your existing presentation template."
        )

    with col2:
        transcript = st.text_area(
            "Paste Meeting Transcript Here",
            height=300,
            help="The AI will use this text to update your slides."
        )

    st.markdown("---")

    # 2. Mode Selection (Simplified for Web)
    mode = st.radio(
        "Select Editing Mode:",
        options=[
            "Batch Edit & Apply (Recommended)",
            "AI Guided Review",
        ],
        horizontal=True,
        index=0,
        key='edit_mode'
    )

    # 3. Execution Button
    if st.button("Start Analysis and Editing 🚀", disabled=(not uploaded_file or not transcript or not API_KEY)):

        # Load the presentation file into a buffer
        try:
            prs = Presentation(io.BytesIO(uploaded_file.read()))
        except Exception as e:
            st.error(f"Could not load presentation: {e}")
            return

        # Extract text blocks and map to shapes
        presentation_blocks, shape_map = extract_presentation_text_blocks(prs)
        if not presentation_blocks:
            st.warning("No editable text blocks found in the presentation. Please check your template.")
            return

        # Generate AI Mapping (Cached)
        # Note: We pass the API key explicitly here to break the cache if the key changes, though st.cache_data is on API key in the function args
        ai_mapping_list = generate_ai_mapping(transcript, presentation_blocks, API_KEY)
        ai_replacements = {item['original_index']: item['new_text'] for item in ai_mapping_list}

        st.session_state['prs'] = prs
        st.session_state['presentation_blocks'] = presentation_blocks
        st.session_state['shape_map'] = shape_map
        st.session_state['ai_replacements'] = ai_replacements
        st.session_state['editing_started'] = True

        # Force a rerun to show the editing interface below
        st.rerun()

    # --- Editing Interface (Runs after analysis is complete) ---
    if 'editing_started' in st.session_state and st.session_state['editing_started']:

        prs = st.session_state['prs']
        presentation_blocks = st.session_state['presentation_blocks']
        ai_replacements = st.session_state['ai_replacements']
        shape_map = st.session_state['shape_map']

        st.header(f"Editing Interface ({mode})")
        st.info(f"AI suggested **{len(ai_replacements)}** blocks for replacement.")

        # Batch Edit & Apply Mode (Mode 3 logic)
        if mode == "Batch Edit & Apply (Recommended)":

            # Generate the Comprehensive Edit Block
            edit_block_output = []
            for block in presentation_blocks:
                original_index = block['id']
                slide_num = block['slide_number']

                if original_index in ai_replacements:
                    initial_text = ai_replacements[original_index].replace("\\n", "\n")
                    indicator = "AI_SUGGESTED"
                else:
                    initial_text = block['current_text']
                    indicator = "ORIGINAL_TEXT"

                # Using ' | ' delimiter for easy parsing
                formatted_line = f"[{original_index}|S{slide_num}|{indicator}] | {initial_text.replace('\n', ' \\n ')}"
                edit_block_output.append(formatted_line)

            st.markdown(
                """
                ### 📝 Step 1: Bulk Edit
                Copy the entire text block below. Edit **ONLY** the text content after the last `|` delimiter. 
                Use `\\n` for new lines or bullet points.
                """
            )

            edited_block_text = st.text_area(
                "Copy, Edit, and Paste Back:",
                value="\n".join(edit_block_output),
                height=600,
                key='batch_edit_area'
            )

            if st.button("Apply Batch Changes and Generate File"):
                edited_block_lines = edited_block_text.split('\n')
                with st.spinner("Applying changes..."):
                    applied_count = process_batch_edits(prs, edited_block_lines, presentation_blocks, shape_map)

                # Save to a bytes buffer
                output_buffer = io.BytesIO()
                prs.save(output_buffer)
                output_buffer.seek(0)

                st.success(f"✅ Success! Applied {applied_count} changes. Your file is ready for download.")
                st.download_button(
                    label=f"Download {OUTPUT_FILENAME}",
                    data=output_buffer,
                    file_name=OUTPUT_FILENAME,
                    mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
                )

        # AI Guided Review Mode (Mode 2 logic)
        elif mode == "AI Guided Review":
            st.warning("Interactive review logic is complex for Streamlit's stateless nature. Please use the Batch Edit mode, which provides maximum control in a web environment.")

# Run the main app function
if __name__ == "__main__":
    main_app()
