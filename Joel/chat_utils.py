import ollama
from rag_utils import retrieve_relevant_chunks
from config import FIXED_SYSTEM_INSTRUCTION, COLOR_BOT, COLOR_WARN, COLOR_INFO
from pdf_utils import CHAT_HISTORY

def stream_response(user_query, model_name):
    global CHAT_HISTORY

    rag_context = retrieve_relevant_chunks(user_query)
    system_instruction = FIXED_SYSTEM_INSTRUCTION + "\n\n" + rag_context

    if not CHAT_HISTORY:
        CHAT_HISTORY.append({"role": "system", "content": system_instruction})
    else:
        CHAT_HISTORY[0]["content"] = system_instruction

    CHAT_HISTORY.append({"role": "user", "content": user_query})

    try:
        stream = ollama.chat(model=model_name, messages=CHAT_HISTORY, stream=True)
        assistant_reply = ""

        print(COLOR_BOT + "Joel: ", end="", flush=True)
        for chunk in stream:
            text = chunk["message"]["content"]
            assistant_reply += text
            print(COLOR_INFO + text, end="", flush=True)

        print("\n")
        CHAT_HISTORY.append({"role": "assistant", "content": assistant_reply})
    except Exception as e:
        CHAT_HISTORY.pop()
        print(COLOR_WARN + f"\n[Streaming Error] {e}\n")
