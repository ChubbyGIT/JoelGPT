import requests
import ollama

def wikipedia_lookup(topic: str) -> str:
    """
    Searches Wikipedia → fetches full article → summarizes using LLM.
    """

    # Step 1 — Search for closest page
    search_url = "https://en.wikipedia.org/w/api.php"
    search_params = {
        "action": "query",
        "list": "search",
        "srsearch": topic,
        "format": "json"
    }

    try:
        search_res = requests.get(search_url, params=search_params).json()
        hits = search_res.get("query", {}).get("search", [])
        if not hits:
            return f"❌ No Wikipedia results for '{topic}'."

        page_title = hits[0]["title"]  # best match
    except Exception as e:
        return f"❌ Wikipedia search failed: {e}"

    # Step 2 — Get full extract of the matched page
    extract_params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": True,
        "titles": page_title,
        "format": "json",
        "redirects": 1
    }

    try:
        extract_res = requests.get(search_url, params=extract_params).json()
        pages = extract_res.get("query", {}).get("pages", {})
        page = next(iter(pages.values()))
        content = page.get("extract", "")

        if not content.strip():
            return f"❌ Could not extract Wikipedia article for '{topic}'."

    except Exception as e:
        return f"❌ Failed to extract article: {e}"

    # Step 3 — Summarize using LLM
    prompt = (
        f"Summarize the following Wikipedia article in clean bullet points.\n"
        f"Topic: {page_title}\n\n"
        f"Article:\n{content}"
    )

    try:
        response = ollama.chat(
            model="gemma3:12b",
            messages=[
                {"role": "system", "content": "You summarize text cleanly and accurately."},
                {"role": "user", "content": prompt}
            ]
        )
        summary = response["message"]["content"]
    except Exception as e:
        return f"❌ LLM summarization failed: {e}"

    return f"📘 **Summary of {page_title}:**\n\n{summary}"
