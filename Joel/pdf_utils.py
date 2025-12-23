# In pdf_utils.py (FINAL FIXED VERSION)

import os
import shutil
import pypdf
from colorama import Fore
import chromadb
# Import types for the Embedding Function
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings 
import ollama 
from config import PDF_FOLDER, CHROMA_COLLECTION as CHROMA_NAME, EMBEDDING_MODEL

# =================================================================
# CRITICAL FIX 1: Explicitly define the Ollama Client and Host
# =================================================================
OLLAMA_HOST = 'http://127.0.0.1:11434'
OLLAMA_CLIENT = ollama.Client(host=OLLAMA_HOST) 
# =================================================================

CHAT_HISTORY = []
# =================================================================
# CRITICAL FIX 2: Switched to PersistentClient
# If you are running this locally, this will store your vector data in the 
# './chroma_db' folder and the data will survive restarts.
# =================================================================
CHROMA_CLIENT = chromadb.PersistentClient(path="./chroma_db") 
CHROMA_COLLECTION = None # Placeholder for the collection object

# ----------------------------------------------------
# Custom Chroma Embedding Function using Ollama
# ----------------------------------------------------
class OllamaEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str):
        self._model_name = model_name
        self.ollama_client = OLLAMA_CLIENT # Use the global explicit client

    def __call__(self, texts: Documents) -> Embeddings:
        embeddings = []
        for text in texts:
            try:
                # Use the explicit client object for the embedding request
                response = self.ollama_client.embeddings(model=self._model_name, prompt=text) 
                embeddings.append(response["embedding"])
            except Exception as e:
                print(Fore.RED + f"Error generating Ollama embedding: {e}")
                raise e 
        return embeddings

# ----------------------------------------------------
# Getter function to safely retrieve the collection
# ----------------------------------------------------
def get_chroma_collection():
    """Returns the initialized Chroma collection object."""
    return CHROMA_COLLECTION
# ----------------------------------------------------
    
def split_text_into_chunks(text, chunk_size=1000, overlap=200):
    """A simple fixed-size text splitter."""
    
    # Split by double newlines for paragraph-like chunks
    text_parts = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    # Simple chunking logic
    chunks = []
    current_chunk = ""
    for part in text_parts:
        if len(current_chunk) + len(part) + 2 < chunk_size:
            current_chunk += part + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = part + "\n\n"
            
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

# ----------------------------------------------------
# NEW FUNCTION: Adds only a single PDF's content (FIXED)
# ----------------------------------------------------
def _add_single_pdf_to_context(path, filename, doc_id_start):
    """Handles PDF parsing, chunking, and addition for a single file."""
    
    documents_to_add = []
    metadatas_to_add = []
    ids_to_add = []
    
    # CRITICAL FIX 3: Start ID counter from 0 for a given file name, 
    # and use the filename + index to generate a truly unique ID.
    chunk_index = 0
    
    try:
        with open(path, "rb") as f:
            reader = pypdf.PdfReader(f)
            
            # CRITICAL FIX 4: Loop through pages and chunk the text page-by-page
            for page_num, page in enumerate(reader.pages):
                text_content = page.extract_text() or ""
                
                # Chunk the text of this single page
                chunks = split_text_into_chunks(text_content) 
                
                for chunk in chunks:
                    if chunk.strip():
                        documents_to_add.append(chunk)
                        # Metadata now correctly reflects the page number (1-indexed)
                        metadatas_to_add.append({"source": filename, "page": page_num + 1}) 
                        # Unique ID for the chunk
                        ids_to_add.append(f"{filename.replace('.pdf', '')}_{chunk_index}") 
                        chunk_index += 1
            
    except Exception as e:
        print(Fore.RED + f"Error loading {filename}: {e}")
        return 0, 0 # Return 0 chunks added

    # Step 3: Embed and Store in Chroma
    if documents_to_add:
        print(Fore.YELLOW + f"Embedding and adding {len(documents_to_add)} chunks from {filename} to Chroma...")
        
        try:
            CHROMA_COLLECTION.add(
                documents=documents_to_add,
                metadatas=metadatas_to_add,
                ids=ids_to_add
            )
            print(Fore.GREEN + f"Successfully stored {len(documents_to_add)} chunks.")
            return len(documents_to_add), chunk_index
        except Exception as e:
            # Catch duplicate ID errors that might occur on re-indexing
            if "already exists" in str(e):
                print(Fore.YELLOW + f"Warning: Chunks for {filename} already exist. Skipping re-add.")
                return len(documents_to_add), chunk_index # Treat as added to avoid error
            print(Fore.RED + f"Error adding documents to Chroma: {e}")
            return 0, 0
    
    return 0, 0

# ----------------------------------------------------
# UPDATED FUNCTION: Handles collection initialization and clear logic
# ----------------------------------------------------
def load_pdfs_into_context(pdf_folder=PDF_FOLDER, clear_existing=True):
    """
    Loads all PDFs in the folder into the Chroma context.
    If clear_existing is True, it clears the collection first (default for main.py startup).
    """
    global CHROMA_COLLECTION
    
    ollama_ef = OllamaEmbeddingFunction(model_name=EMBEDDING_MODEL)

    # Step 1: Handle Collection Initialization and Clearing
    print(Fore.YELLOW + "Initializing ChromaDB...")
    
    if clear_existing:
        print(Fore.YELLOW + f"Clearing existing Chroma context (Wipe & Re-create collection '{CHROMA_NAME}')...")
        try:
            # FIX: Robust wipe by deleting the collection via the client.
            CHROMA_CLIENT.delete_collection(name=CHROMA_NAME)
        except Exception as e:
             # Ignore the error if the collection didn't exist
            if "not found" not in str(e) and "does not exist" not in str(e) and "already deleted" not in str(e):
                print(Fore.RED + f"Error during collection delete: {e}")
            
        # Re-create the collection
        CHROMA_COLLECTION = CHROMA_CLIENT.get_or_create_collection(
            name=CHROMA_NAME,
            embedding_function=ollama_ef 
        )
        print(Fore.GREEN + f"ChromaDB collection '{CHROMA_NAME}' re-created and ready.")
        
    elif CHROMA_COLLECTION is None: # Standard initialization if not clearing
        CHROMA_COLLECTION = CHROMA_CLIENT.get_or_create_collection(
            name=CHROMA_NAME,
            embedding_function=ollama_ef
        )
        print(Fore.GREEN + f"ChromaDB collection '{CHROMA_NAME}' ready.")


    # Step 2: Read PDFs and chunk text
    if not os.path.isdir(pdf_folder):
        os.makedirs(pdf_folder)

    print(Fore.YELLOW + f"Loading PDFs from '{pdf_folder}'...")
    pdf_count = 0
    total_chunks = 0
    
    for filename in os.listdir(pdf_folder):
        if filename.lower().endswith(".pdf"):
            pdf_count += 1
            path = os.path.join(pdf_folder, filename)
            
            # Pass 0 as the starting ID, it will be ignored by the fixed function
            chunks_added, _ = _add_single_pdf_to_context(path, filename, 0) 
            total_chunks += chunks_added

    print(Fore.GREEN + f"Successfully processed {pdf_count} PDF(s). Total chunks stored: {total_chunks}")
    return "Vector context loaded."

# ----------------------------------------------------
# Incremental Upload Function (FIXED FOR NEW ID LOGIC)
# ----------------------------------------------------
def handle_upload():
    global CHAT_HISTORY
    # Ensure CHROMA_COLLECTION is initialized before trying to use it
    if CHROMA_COLLECTION is None:
        load_pdfs_into_context(clear_existing=False) # Ensure collection is created if not already

    print(Fore.YELLOW + "\n--- PDF UPLOAD MODE ---")
    source_path = input(Fore.CYAN + "Enter full path to PDF: ").strip().replace('"', "").replace("'", "")

    if not source_path:
        print(Fore.YELLOW + "Upload cancelled.")
        return
    if not os.path.exists(source_path) or not source_path.lower().endswith(".pdf"):
        print(Fore.RED + "Invalid PDF path.")
        return

    filename = os.path.basename(source_path)
    dest = os.path.join(PDF_FOLDER, filename)
    
    if os.path.exists(dest):
        print(Fore.YELLOW + f"Warning: '{filename}' already exists. Overwriting...")

    try:
        shutil.copy(source_path, dest)
        print(Fore.GREEN + f"Copied '{filename}' to {PDF_FOLDER}.")
        
        # --- Incremental Logic: Only add the newly uploaded file ---
        # The start ID passed here is now irrelevant, but we pass 0 for consistency
        _add_single_pdf_to_context(dest, filename, 0) 
        # ----------------------------------------------------
        
        print(Fore.YELLOW + "RAG context refreshed.")
        CHAT_HISTORY = []
        print(Fore.YELLOW + "Chat history cleared.")
    except Exception as e:
        print(Fore.RED + f"Copy or indexing error: {e}")