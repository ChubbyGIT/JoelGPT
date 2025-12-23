import sys
from colorama import init
init(autoreset=True)
import chromadb

from config import MODEL_NAME
# --- CRITICAL FIX: Updated import for web_search_lookup ---
# The wikipedia_lookup is no longer needed in this file
from ollama_utils import ensure_ollama_running, web_search_lookup
from pdf_utils import handle_upload, load_pdfs_into_context
from chat_utils import stream_response
from input_utils import get_multiline_input
# from wikipedia_lookup import wikipedia_lookup   # <-- REMOVED THIS IMPORT

# These functions can cause the program to hang if the server/db fails
ensure_ollama_running()
load_pdfs_into_context()

def run_chat():
    print("🤖 Joel AI Assistant Initializing...")

    while True:
        start = input("Start Chat (type 'hey joel'): ").strip().lower()
        if start == "hey joel":
            break
        if start == "/exit":
            print("Goodbye 👋")
            sys.exit(0)

    print("\nJoel is ready! Ask questions or use /upload to add PDFs.\n")

    while True:
        try:
            user_input = get_multiline_input()
        except KeyboardInterrupt:
            user_input = "/exit"

        # Exit command
        if user_input.lower() == "/exit":
            print("Goodbye 👋")
            break

        # Upload PDF
        if user_input.lower() == "/upload":
            handle_upload()
            continue

        # Real-time Web Search Command (Fixed /look function)
        if user_input.lower().startswith("/search "):
            query = user_input[8:].strip()
            if not query:
                # Updated hint to reflect the new command's capability
                print("❌ Please provide a query: /search latest rcb win")
                continue

            # This calls the new function in ollama_utils.py
            results = web_search_lookup(query)
            print(results + "\n")
            continue

        # Regular conversation
        if user_input.strip():
            stream_response(user_input, MODEL_NAME)


if __name__ == "__main__":
    run_chat()