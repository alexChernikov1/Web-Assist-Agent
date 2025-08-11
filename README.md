# Web-Assist Agent 🛠️

A FastAPI-powered appliance-parts Q&A service that combines:

* **LangGraph** (React-style, tool-calling agent)
* **DeepSeek** & **OpenAI** LLMs  
* **Firecrawl** and **Selenium** for scraping
* JSON & disk-based caches to stay fast across sessions

---

## ✨ Key Features
* **Model ↔ Part compatibility** look-ups with cached cross-references  
* **Fallback Firecrawl research** for open-ended questions  
* **Automatic context filling** when the user says “this model” or “that part”  
* **Flow-controlled scraping** (JSON API → visible Firefox → inner-div scrolling)  
* **Debug-friendly logging** (tool calls, planner prompts, cache hits)

---

## 🔐 Environment Variables

| Name | Purpose |
|------|---------|
| `DEEP_SEEK_WEB_API_KEY` | DeepSeek LLM access |
| `OPENAI_API_KEY` | OpenAI (used by LangGraph planner) |
| `FIRECRAWL_API_KEY` | Firecrawl site-scraper |

Create a `.env` (or `python-dotenv.env`) with the three keys.

---

## 🗂️ Install & Run

```bash
# 1. create / activate a venv
python -m venv .venv
. .venv/Scripts/Activate      # PowerShell on Windows
# 2. install deps
pip install -r requirements.txt
# 3. run API
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive Swagger UI.

---

## 🔄 High-Level Flow

```txt
A[User /chat POST] --> B[analyse(q)]
B -->|compat_intent|   C[Pre-warm model-part cache] --> D[Build agent messages] |
B -->|no_compat_intent|            F[Skip pre-warm] --> D[Build agent messages] |
                                                                                |
                                                                                |
      E[LangGraph agent] <------------------------------------------------------┘
          |
          |  ┌──────────────────────── Tool Calls (loop) ────────────────────────┐
          |  | E -->|scrape_model()|      G[scrape_model_tool]        --> E      |
          |  | E -->|scrape_part()|       H[scrape_part_tool]         --> E      |
          |  | E -->|get_compat()|        I[get_compat_tool]          --> E      |
          |  | E -->|firecrawl()|         J[firecrawl_research_tool]  --> E      |
          |  | E -->|need_ctx()|          K[need_more_context_yes_no] --> E      |
          |  | E -->|fill_ctx()|          L[fill_context_tool]        --> E      |
          |  | E -->|compat_reason()|     M[compat_reasoning_tool]    --> E      |
          |  | E -->|part_lookup()|       N[part_lookup_tool]         --> E      |
          |  | E -->|model_lookup()|      O[model_lookup_tool]        --> E      |
          |  | E -->|session_cache()|     P[session_cache_tool]       --> E      |
          |  └───────────────────────────────────────────────────────────────────┘
          |      ↑                                                        ↓
          |      └────────── repeats until all context is ready ──────────┘
          v
      Q[Final LLM answer]
          |
          v
      R[Return JSON { reply: … }]
```


---

## 📝 API Endpoint

| Method | Path | Body schema | Description |
|--------|------|-------------|-------------|
| `POST` | `/chat` | `{ "message": "string" }` | Returns `{ "reply": "string" }` |

---

## 📁 Cache Files

| File | What it stores |
|------|----------------|
| `model_compat_cache.json` | Model → compatible parts |
| `part_compat_cache.json`  | Part → cross-ref models |
| `model_page_cache.json`   | Raw HTML for model pages |
| `part_page_cache.json`    | Raw HTML for part pages |
| `visited_links.json`      | URLs explored by Firecrawl agent |

All caches are auto-created and updated; delete them to force fresh scraping.

---

## 🧪 Local Smoke Test

```bash
curl -X POST http://127.0.0.1:8000/chat ^
     -H "Content-Type: application/json" ^
     -d "{"message": "Is part PS11752778 compatible with model KUDS01FLSS0?"}"
```

---

## 🙏 Credits

* **PartSelect.com** for the public data.
* **LangChain / LangGraph** for the planning framework.
* **DeepSeek & OpenAI** for the language models.
* **Firecrawl** for resilient scraping.

---

> _“Think step-by-step, decide which tool to call, and keep calling tools until you can answer the user.”_ – System Prompt
