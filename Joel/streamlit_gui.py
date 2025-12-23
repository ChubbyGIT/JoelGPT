import streamlit as st
import ollama
import sys
import os
import re
from io import BytesIO

# --- Import from local project files ---
from config import MODEL_NAME, FIXED_SYSTEM_INSTRUCTION
# Note: We must import the OLLAMA_CLIENT from pdf_utils to ensure consistency
from pdf_utils import load_pdfs_into_context, CHAT_HISTORY, OLLAMA_CLIENT, PDF_FOLDER, get_chroma_collection
from rag_utils import retrieve_relevant_chunks
from ollama_utils import ensure_ollama_running, web_search_lookup 
# --- End Imports ---

# -----------------
# 1. SETUP & THEME (CSS for Alignment and Color)
# -----------------
st.set_page_config(
    page_title="Joel AI Assistant",
    layout="wide"
)

# Apply Black and Gold Yellow Theme and Chat Alignment via Custom CSS
st.markdown("""
<style>
/* Main Background (Black) */
.stApp {
    background-color: #000000;
}
/* Chat Message Bubbles (Dark Grey) */
.stChatMessage {
    background-color: #1e1e1e; 
    border-radius: 8px;
}
/* Joel's Messages (Gold Yellow text on dark background - Left Aligned) */
.stChatMessage[data-testid="stChatMessage"] {
    background-color: #1e1e1e;
    text-align: left;
}
.stChatMessage[data-testid="stChatMessage"] .stMarkdown > div {
    color: #FFD700; /* Gold Yellow for text */
}

/* User Messages (Light Grey text - RIGHT ALIGNED FIX) */
.stChatMessage[data-testid="stChatMessageUser"] {
    background-color: #333333; /* Slightly lighter dark color for distinction */
    text-align: right; /* Align bubble content to the right */
}
.stChatMessage[data-testid="stChatMessageUser"] .stMarkdown > div {
    color: #C0C0C0; /* Light Grey/Silver for user text */
}
/* Ensure the markdown content itself is pushed to the right within the user bubble */
.stChatMessage[data-testid="stChatMessageUser"] .stMarkdown {
    display: flex;
    justify-content: flex-end;
}

/* Input Container Background */
[data-testid="stChatInput"] {
    background-color: #1e1e1e;
    border-top: 1px solid #FFD700; /* Gold separator line */
}
/* Input Text Box */
[data-testid="stChatInput"] textarea {
    color: #FFD700; /* Gold Yellow text input */
    background-color: #303030;
}
/* Status text */
.stProgress > div > div > div > div {
    background-color: #FFD700;
}
/* Stop Button styling */
.stop-button-container {
    display: flex;
    justify-content: center;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("🤖 Joel AI Assistant")
st.markdown("---")

# -----------------
# 2. STATE MANAGEMENT & INITIALIZATION
# -----------------
if "is_generating" not in st.session_state:
    st.session_state.is_generating = False 
if "stop_generation" not in st.session_state:
    st.session_state.stop_generation = False 
if "current_prompt" not in st.session_state:
    st.session_state.current_prompt = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] 
    st.session_state.chat_history.append({"role": "assistant", "content": 
        "Hello! I am Joel, your RAG-enabled AI assistant. "
        "Use the sidebar to upload PDFs or view the available documents."
    })


@st.cache_resource
def initialize_environment():
    """Run environment checks and RAG context loading once."""
    try:
        ensure_ollama_running() 
        # Load context. This call also initializes the CHROMA_COLLECTION object globally.
        load_pdfs_into_context(clear_existing=False) 
        return True
    except Exception as e:
        st.error(f"Initialization Failed: {e}. Please ensure Ollama is installed and running.")
        return False

# Display initialization status
with st.spinner('Initializing Joel AI Assistant and loading RAG context...'):
    if initialize_environment():
        pass

# -----------------
# 3. CORE HANDLERS
# -----------------

def handle_stop_click():
    """Sets the flag to stop the streaming process and resets the generation state."""
    st.session_state.stop_generation = True
    st.session_state.is_generating = False 
    st.session_state.current_prompt = None
    st.warning("❌ Generation stopped by user. Re-enabling chat input.")

def _add_pdf_to_rag(file_name: str, file_bytes: bytes):
    """Saves the uploaded file and triggers RAG re-indexing."""
    # Import necessary functions from pdf_utils locally
    from pdf_utils import _add_single_pdf_to_context, PDF_FOLDER, CHAT_HISTORY 
    
    try:
        pdf_folder = PDF_FOLDER 
        os.makedirs(pdf_folder, exist_ok=True)
        dest_path = os.path.join(pdf_folder, file_name)

        # 1. Save the file
        with open(dest_path, "wb") as f:
            f.write(file_bytes)

        # 2. Index the new file
        _add_single_pdf_to_context(dest_path, file_name, 0)
        
        # 3. Update Streamlit and Global History
        st.session_state.chat_history.append({"role": "assistant", 
                                              "content": f"✅ PDF **'{file_name}'** uploaded and RAG context updated."})
        CHAT_HISTORY.clear() # Clear the global model history
        
    except Exception as e:
        st.session_state.chat_history.append({"role": "assistant", 
                                              "content": f"❌ PDF Upload Error for **'{file_name}'**: {e}"})
        st.error(f"Error processing PDF: {e}")

def stream_response_generator(user_query):
    # ... (stream_response_generator remains the same as the previous version) ...
    """
    Generator that handles RAG, Ollama chat, and checks for the stop signal.
    """
    global CHAT_HISTORY

    # 1. RAG Context Retrieval (blocking)
    rag_context = retrieve_relevant_chunks(user_query)
    system_instruction = FIXED_SYSTEM_INSTRUCTION + "\n\n" + rag_context

    # Update Global History
    if not CHAT_HISTORY:
        CHAT_HISTORY.append({"role": "system", "content": system_instruction})
    else:
        CHAT_HISTORY[0]["content"] = system_instruction
        
    CHAT_HISTORY.append({"role": "user", "content": user_query})

    # 2. Stream from Ollama
    assistant_reply = ""
    try:
        stream = OLLAMA_CLIENT.chat(model=MODEL_NAME, messages=CHAT_HISTORY, stream=True)
        
        for chunk in stream:
            if st.session_state.stop_generation:
                break 
                
            text = chunk["message"]["content"]
            assistant_reply += text
            yield text
            
        # 3. Final Update to global CHAT_HISTORY
        if assistant_reply.strip() and not st.session_state.stop_generation:
            CHAT_HISTORY.append({"role": "assistant", "content": assistant_reply})
            
        elif st.session_state.stop_generation:
            truncated_reply = assistant_reply + "\n\n**[Response stopped by user]**"
            CHAT_HISTORY.append({"role": "assistant", "content": truncated_reply})
            st.session_state.stop_generation = False 
            yield "\n\n**[Response stopped by user]**"
            
    except Exception as e:
        if CHAT_HISTORY and CHAT_HISTORY[-1]["role"] == "user":
            CHAT_HISTORY.pop()
        error_msg = f"**An error occurred:** {e}. Please check your Ollama server and model '{MODEL_NAME}'."
        yield error_msg


def handle_input_submit():
    """
    Callback function that runs when the chat input is submitted.
    This sets the 'is_generating' state and queues the prompt *immediately*.
    """
    if st.session_state.chat_input_widget:
        prompt = st.session_state.chat_input_widget
        st.session_state.current_prompt = prompt
        st.session_state.is_generating = True
        st.session_state.stop_generation = False
        st.session_state.chat_input_widget = ""
        st.rerun() 

# -----------------
# 4. SIDEBAR (File Uploader & Viewer)
# -----------------
with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader(
        "Choose a PDF file to add to Joel's knowledge base:",
        type="pdf",
        accept_multiple_files=False,
        key="pdf_uploader",
        disabled=st.session_state.is_generating
    )
    
    if uploaded_file is not None:
        if st.button("Add PDF to RAG Context", use_container_width=True, disabled=st.session_state.is_generating):
            with st.spinner(f"Processing '{uploaded_file.name}' and updating RAG context..."):
                _add_pdf_to_rag(uploaded_file.name, uploaded_file.read())
            st.rerun()

    st.markdown("---")
    st.header("Available RAG Documents")
    
    try:
        # Get the ChromaDB collection object
        collection = get_chroma_collection()
        
        if collection and collection.count() > 0:
            # Retrieve unique document sources (file names) from metadata
            # This is a bit complex in Chroma without direct 'DISTINCT' queries, 
            # so we fetch all metadatas and find unique sources.
            all_metadatas = collection.get(include=['metadatas'])['metadatas']
            unique_sources = set(m.get('source') for m in all_metadatas if m and m.get('source'))
            
            if unique_sources:
                st.info(f"Found {len(unique_sources)} indexed documents:")
                for source_name in sorted(list(unique_sources)):
                    pdf_path = os.path.join(PDF_FOLDER, source_name)
                    
                    if os.path.exists(pdf_path):
                        # Read the file bytes
                        with open(pdf_path, "rb") as f:
                            pdf_bytes = f.read()
                        
                        # Use download_button as the Streamlit-native way to prompt view/download
                        st.download_button(
                            label=f"📄 {source_name}",
                            data=pdf_bytes,
                            file_name=source_name,
                            mime='application/pdf',
                            key=f"download_{source_name}",
                            use_container_width=True
                        )
                    else:
                        st.caption(f"⚠️ {source_name} (File not found)")
            else:
                st.caption("No unique sources found in the database.")
        else:
            st.caption("The RAG database is currently empty.")
            
    except Exception as e:
        st.error(f"Error listing documents: {e}")
        st.caption("Ensure Ollama and ChromaDB are running correctly.")


# -----------------
# 5. DISPLAY HISTORY & INPUT
# -----------------

# Display chat history first
chat_placeholder = st.container()
with chat_placeholder:
    for message in st.session_state.chat_history:
        # Determine the avatar for both the user and assistant
        avatar = "👤" if message["role"] == "user" else "🤖"
        # Determine the user message flag for CSS alignment
        is_user = message["role"] == "user"
        
        # Display the message
        with st.chat_message(message["role"], avatar=avatar):
            # The CSS in the <style> block handles the right alignment for 'user' role
            st.markdown(message["content"])

# Place the stop button above the chat input, immediately when generating is True
if st.session_state.is_generating:
    st.markdown('<div class="stop-button-container">', unsafe_allow_html=True)
    st.button("🔴 Stop Generation", on_click=handle_stop_click, type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

# Handle user input via on_submit callback for immediate state change
st.chat_input(
    "Ask Joel a question or use /search <query>",
    disabled=st.session_state.is_generating,
    key="chat_input_widget", 
    on_submit=handle_input_submit
)

# -----------------
# 6. GENERATION LOGIC (Runs only when a prompt is queued)
# -----------------
if st.session_state.current_prompt:
    
    prompt_to_process = st.session_state.current_prompt
    st.session_state.current_prompt = None 
    
    # --- 1. Display User Message (redundant as it's handled by history, but useful for immediate view) ---
    st.session_state.chat_history.append({"role": "user", "content": prompt_to_process})
    
    # Display the user message immediately in the placeholder
    with chat_placeholder:
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt_to_process)

    # --- 2. Process Command or Stream Response ---
    
    is_command = prompt_to_process.lower().startswith("/search ")
    
    if is_command:
        query = prompt_to_process[8:].strip()
        
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner(f"Searching the web for '{query}'..."):
                results_text_raw = web_search_lookup(query)
                results_text = re.sub(r'\x1b\[[0-9;]*m', '', results_text_raw).replace('\n', '\n\n')
                st.markdown(results_text)
                
            st.session_state.chat_history.append({"role": "assistant", "content": results_text})
            
    else:
        with st.chat_message("assistant", avatar="🤖"):
            response_generator = stream_response_generator(prompt_to_process)
            full_assistant_response = st.write_stream(response_generator)
            st.session_state.chat_history.append({"role": "assistant", "content": full_assistant_response})

    # Final cleanup: Reset is_generating state and rerun
    st.session_state.is_generating = False 
    st.rerun()