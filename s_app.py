import ollama
import sys
import os
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
import pypdf
import pyttsx3

# --- Config ---
FIXED_SYSTEM_INSTRUCTION = (
    "You are 'Joel', a helpful, professional, and highly capable AI assistant. "
    "Your goal is to provide accurate, clear, and concise answers. "
    "Perform tasks immediately. Be polite and efficient. "
    "If asked to search, summarize information from the web clearly with sources."
)

MODEL_NAME = "llama3.1:latest"

# Initialize TTS engine
tts = pyttsx3.init()

# --- RAG Core Utilities ---
def load_pdfs_into_context(pdf_folder="data_pdfs"):
    context = ""
    if not os.path.isdir(pdf_folder):
        return context

    for filename in os.listdir(pdf_folder):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(pdf_folder, filename)
            try:
                with open(path, "rb") as file:
                    reader = pypdf.PdfReader(file)
                    for page in reader.pages:
                        context += page.extract_text() + "\n"
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    return context

DOCUMENT_CONTEXT = load_pdfs_into_context()


# --- RAG retrieval ---
def retrieve_relevant_chunks(user_query: str, top_k=3):
    if not DOCUMENT_CONTEXT.strip():
        return ""

    query_words = set(user_query.lower().split())
    chunks = DOCUMENT_CONTEXT.split("\n")

    ranked = []
    for chunk in chunks:
        score = sum(1 for word in query_words if word in chunk.lower())
        if score > 0:
            ranked.append((score, chunk))

    ranked.sort(reverse=True, key=lambda x: x[0])
    return "\n".join(chunk for score, chunk in ranked[:top_k])


# --- LLM Response ---
def generate_response(user_query, use_web=False, web_content=None):
    rag_response = retrieve_relevant_chunks(user_query)
    system_instruction = FIXED_SYSTEM_INSTRUCTION
    if use_web and web_content:
        system_instruction += "\n\nUse the following web information:\n" + web_content

    prompt = (
        f"{system_instruction}\n\n"
        f"Relevant Context:\n{rag_response}\n\n"
        f"User Query: {user_query}\n\n"
        f"Final Answer:"
    )

    try:
        response = ollama.generate(
            model=MODEL_NAME,
            prompt=prompt
        )
        return response["response"]
    except Exception as e:
        return f"Error generating response: {e}"


# --- Web Search + Scraping ---
def search_web(query: str, result_count=3):
    results_text = ""
    try:
        print(f"🔍 Searching the web for: {query}")
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=result_count)

        for r in results:
            url = r.get("href")
            if not url:
                continue

            page_content = scrape_page(url)
            if page_content:
                results_text += f"Source: {url}\n{page_content}\n\n"

    except Exception as e:
        results_text += f"Web search failed: {e}\n"

    return results_text


def scrape_page(url: str):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=6)
        soup = BeautifulSoup(res.text, "html.parser")

        # Extract text from <p>
        text = " ".join([p.get_text() for p in soup.find_all("p")])
        return " ".join(text.split()[:200])  # limit to first 200 words
    except:
        return None


# --- TTS Output ---
def speak_text(text: str):
    tts.say(text)
    tts.runAndWait()


# --- Main App Logic ---
def run_chat():
    print("👋 Joel is ready! Type /exit to quit.")
    use_voice = input("Enable voice? (y/n): ").strip().lower() == "y"

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() == "/exit":
            print("Goodbye 👋")
            break

        # Detect web search command
        if user_input.startswith("/search"):
            query = user_input.replace("/search", "").strip()
            web_data = search_web(query)
            output = generate_response(query, use_web=True, web_content=web_data)

        else:
            output = generate_response(user_input)

        print(f"\nJoel: {output}\n")

        if use_voice:
            speak_text(output)


if __name__ == "__main__":
    run_chat()
