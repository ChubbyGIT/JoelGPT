from colorama import Fore

MODEL_NAME = "gemma3:12b"
# --- New Constants for VectorDB ---
EMBEDDING_MODEL = "all-minilm" # A good choice for Ollama embeddings
CHROMA_COLLECTION = "pdf_rag_chunks"
# ---
PDF_FOLDER = "data_pdfs"

FIXED_SYSTEM_INSTRUCTION = (
    "You are 'Joel', a helpful, professional, and highly capable AI assistant. "
    "You answer clearly and concisely, and you may use uploaded PDF context."
)

# Optional: colors
COLOR_USER = Fore.CYAN
COLOR_BOT = Fore.GREEN
COLOR_WARN = Fore.YELLOW
COLOR_ERROR = Fore.RED
COLOR_INFO = Fore.WHITE
