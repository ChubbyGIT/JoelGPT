# In rag_utils.py (with added diagnostic prints)

import ollama 
# CRITICAL FIX 3: Import the getter function and the client/host from pdf_utils
from pdf_utils import get_chroma_collection, OLLAMA_CLIENT, OLLAMA_HOST
from config import EMBEDDING_MODEL, COLOR_WARN 

def retrieve_relevant_chunks(query, top_k=5):
    """
    Performs a Vector Search using an Ollama embedding model and ChromaDB.
    """
    
    # CRITICAL FIX 4: Get the initialized collection object
    CHROMA_COLLECTION = get_chroma_collection() 
    
    if CHROMA_COLLECTION is None or CHROMA_COLLECTION.count() == 0:
        print(COLOR_WARN + "[RAG] No documents in ChromaDB collection.") # Diagnostic print
        return "No vector context available in ChromaDB."

    try:
        # Step 1: Get the query embedding from Ollama
        print(COLOR_WARN + f"[RAG] Generating embedding for query with {EMBEDDING_MODEL}...")
        
        # CRITICAL FIX 5: Use the explicit OLLAMA_CLIENT
        query_embedding_res = OLLAMA_CLIENT.embeddings(
            model=EMBEDDING_MODEL,
            prompt=query
        )
        query_embedding = query_embedding_res["embedding"]
        
        # Step 2: Query ChromaDB using the embedding
        print(COLOR_WARN + f"[RAG] Querying ChromaDB for top {top_k} matches...")
        
        results = CHROMA_COLLECTION.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Step 3: Format the retrieved context
        context_chunks = []
        if results and results.get('documents') and results['documents'][0]:
            for doc, metadata, distance in zip(
                results['documents'][0], 
                results['metadatas'][0],
                results['distances'][0]
            ):
                context_chunks.append(
                    f"--- Source: {metadata.get('source', 'Unknown')} (Score: {distance:.4f}) ---\n"
                    f"{doc}"
                )

        context = "\n\n".join(context_chunks)
        
        # --- DIAGNOSTIC LOGGING ---
        if context.strip():
            print(COLOR_WARN + f"[RAG] Successfully retrieved {len(context_chunks)} chunks.")
            return context
        else:
            print(COLOR_WARN + "[RAG] No relevant chunks found in ChromaDB.")
            return "No relevant information found."
        # --------------------------

    except Exception as e:
        if "ConnectionError" in str(e) or "Timeout" in str(e):
             return f"Error during vector retrieval: Ollama server or model '{EMBEDDING_MODEL}' is not available on {OLLAMA_HOST}. Please check your Ollama installation."
        print(COLOR_WARN + f"[RAG Error] Vector retrieval failed: {e}")
        return f"Error during vector retrieval: {e}"