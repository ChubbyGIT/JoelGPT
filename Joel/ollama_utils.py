# In ollama_utils.py

import subprocess
import time
import requests
import ollama
from colorama import Fore
# Import MODEL_NAME and COLOR_ variables for the new web_search_lookup function
from config import EMBEDDING_MODEL, MODEL_NAME, COLOR_WARN, COLOR_INFO 
from pdf_utils import OLLAMA_HOST 

def ensure_ollama_running():
    
    # Use the explicit host defined in pdf_utils for consistency
    ollama_url = f"{OLLAMA_HOST}/api/tags" # Should be http://127.0.0.1:11434/api/tags
    
    try:
        # Check 1: Is it already running?
        requests.get(ollama_url, timeout=1)
        # If successful, we skip the rest.
        print(Fore.GREEN + "✅ Ollama server is already running.")
    except:
        # Check 2: If not running, start it
        print(Fore.YELLOW + "⚡ Starting Ollama server in background...")
        # Start Ollama with the host variable set explicitly, matching the Python client
        # Note: This works best if you set OLLAMA_HOST environment variable before running main.py
        # For Popen, we keep it simple, relying on the client fix.
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Wait loop for the server to start (INCREASED RETRIES)
        for i in range(60): # Increased from 40 tries to 60 (30 seconds total)
            try:
                requests.get(ollama_url, timeout=2) # Increased timeout to 2s
                print(Fore.GREEN + "✅ Ollama server is ready.")
                break # Success! Exit the loop
            except:
                time.sleep(0.5)
        else:
            # If the loop finished without breaking (i.e., failed to connect 60 times)
            raise RuntimeError("❌ Failed to start Ollama server. Check Task Manager for rogue processes or firewall.")
    
    # The check for the model (which you already seem to have succeeding)
    # --- Ensure Embedding Model is available ---
    print(Fore.YELLOW + f"⬇️ Checking embedding model '{EMBEDDING_MODEL}' status...")
    try:
        # Use subprocess to check if it's there, pull only if necessary
        subprocess.run(["ollama", "list", EMBEDDING_MODEL], check=True, capture_output=True)
        print(Fore.GREEN + f"✅ Embedding model '{EMBEDDING_MODEL}' is ready.")
    except Exception:
        # If 'ollama list' fails (model not found), pull it.
        print(Fore.YELLOW + f"⬇️ Pulling embedding model '{EMBEDDING_MODEL}'...")
        try:
            subprocess.run(["ollama", "pull", EMBEDDING_MODEL], check=True)
            print(Fore.GREEN + f"✅ Embedding model '{EMBEDDING_MODEL}' successfully pulled.")
        except Exception as e:
            print(Fore.RED + f"❌ Failed to pull embedding model: {e}")
            
# Rerun main.py after updating ollama_utils.py


# In ollama_utils.py

# ... (other code)

def web_search_lookup(query: str) -> str:
    """
    Performs a real-time web search using the Ollama Web Search API.
    """
    # Use COLOR_WARN for initial message
    print(COLOR_WARN + f"[Web Search] Searching the internet for '{query}'...")
    try:
        # --- FIX IS HERE: Remove the 'model' argument ---
        response = ollama.web_search(
            query=query
            # The original code had: model=MODEL_NAME  <-- DELETE THIS LINE
        )
        
        # The rest of the function remains the same...
        summary = response.get("summary", "No summary available.")
        results = response.get("results", [])
        
        # Format sources with title and URL
        formatted_results = [f"- {r.get('title', 'Untitled')} ({r.get('url', 'No URL')})" for r in results]

        output = (
            f"✅ **Web Search Results for '{query}'**\n\n"
            f"**Summary:**\n{summary}\n\n"
            f"**Sources Found:**\n"
            f"{'\\n'.join(formatted_results)}\n"
        )
        # Use COLOR_INFO for the final output
        return COLOR_INFO + output

    except ollama.ResponseError as e:
        # ... (rest of the error handling)
        return COLOR_WARN + f"❌ Ollama Web Search Error: {e}. Ensure your Ollama model is up-to-date and supports web search."
    except Exception as e:
        return COLOR_WARN + f"❌ An unexpected error occurred during web search: {e}"