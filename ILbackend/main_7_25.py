# main.py ─ Firecrawl‑powered FastAPI backend (rate‑limit back‑off + **dual HTML caches**)
import logging, os, re, textwrap, time, json, requests
from typing import Literal, Tuple, List, Dict, Set, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from dotenv import load_dotenv
from firecrawl import FirecrawlApp
from openai import OpenAI

# ───── Logging ─────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn.error")

# ───── env + clients ─────
load_dotenv("python-dotenv.env")
DEEPSEEK_API_KEY = os.getenv("DEEP_SEEK_WEB_API_KEY")
if not DEEPSEEK_API_KEY:
    logger.error("DeepSeek API key not found in DEEP_SEEK_WEB_API_KEY")

client   = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
FC_KEY   = os.getenv("FIRECRAWL_API_KEY")
if not FC_KEY:
    raise RuntimeError("FIRECRAWL_API_KEY missing in environment")
fc_client = FirecrawlApp(api_key=FC_KEY)

# ───── Constants ─────
RATE_LIMIT_SLEEP, MAX_RETRIES = 60, 3 # Sleep time and retry count for Firecrawl rate limiting

COMPAT_CACHE_FILE   = "model_compat_cache.json" # File path for model-to-compatible-parts cache

MODEL_PAGE_FILE     = "model_page_cache.json" # File path for cached HTML of model pages

PART_PAGE_FILE      = "part_page_cache.json" # File path for cached HTML of part pages

VISITED_LINKS_FILE  = "visited_links.json" # File path for storing all successfully visited links           

# ───── Disk caches ─────
def _load_cache(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            logger.info(f"[CACHE] loaded {len(data)} items from {path}")
            return data
    except (FileNotFoundError, json.JSONDecodeError):
        logger.info(f"[CACHE] no existing {path}, starting fresh")
        return default

def _write_cache(path: str, data):
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        logger.info(f"[CACHE] wrote {path}")
    except Exception as e:
        logger.warning(f"[CACHE] failed writing {path}: {e}")

MODEL_COMPAT:      Dict[str, List[str]]     = _load_cache(COMPAT_CACHE_FILE, {})
MODEL_HTML_CACHE:  Dict[str, Dict[str,str]] = _load_cache(MODEL_PAGE_FILE, {})
PART_HTML_CACHE:   Dict[str, Dict[str,str]] = _load_cache(PART_PAGE_FILE , {})
VISITED_LINKS:     List[str]               = _load_cache(VISITED_LINKS_FILE, [])   

# ───── Firecrawl fetch helper ─────
def fetch_html(url: str) -> str:
    for attempt in range(1, MAX_RETRIES + 1):
        logger.info(f"[FIRECRAWL] Scraping {url} (try {attempt}/{MAX_RETRIES})")
        try:
            r = fc_client.scrape_url(url=url, formats=["html"])
            return r.html or ""
        except Exception as e:
            if "429" in str(e):
                logger.warning(f"[FIRECRAWL WAIT] hit 429, sleeping {RATE_LIMIT_SLEEP}s")
                time.sleep(RATE_LIMIT_SLEEP)
                continue
            logger.error(f"[FIRECRAWL ERROR] {e}")
            return ""
    logger.error(f"[FIRECRAWL ERROR] failed after retries")
    return ""

# ───── Runtime state ─────
HISTORY_LIMIT = QA_LIMIT = 20
BLOB_HISTORY: List[str]          = []
QA_HISTORY:   List[Tuple[str,str]] = []
PART_DICT:    Dict[str,str]      = {}
MODEL_DICT:   Dict[str,str]      = {}
# ───── Conversational session state ─────
SESSION_STATE: dict[str, list[str]] = {
    "models": [],   # most‑recent model codes (latest first)
    "parts":  []    # most‑recent part numbers  (latest first)
}
# ───── LLM helper ─────
def ask_llm(sys: str, user: str, timeout: int = 500) -> str:
    try:
        r = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": sys},
                      {"role": "user",   "content": user}],
            timeout=timeout,
            stream=False,
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"[LLM WARNING] {e}")
        return ""

# ───── Regex & extraction ─────
COMPAT_RE = re.compile(r"\b(compatib(?:le|ility)|fit|works?\s+with|match(?:es)?)\b", re.I)
MODEL_PATTERN = re.compile(r"\b([A-Za-z0-9][A-Za-z0-9\-\.]{5,})\b")
BLOCKLIST = {"COMPATIBLE", "MODEL", "PART", "NUMBER"}

# Classify if a question is about compatibility using regex or LLM fallback
def llm_is_compat(q: str) -> Literal["YES", "NO"]:
    if COMPAT_RE.search(q):
        return "YES"
    return "YES" if ask_llm("Return ONLY YES if asking part‑model compatibility, else NO.", q, 120) == "YES" else "NO"

# Extract unique valid model numbers from a string
def extract_model_numbers(t: str) -> List[str]:
    seen, out = set(), []
    for raw in MODEL_PATTERN.findall(t):
        m = raw.rstrip("?.!,")
        if m.upper() in BLOCKLIST or m.isalpha():
            continue
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out

# Extract all unique integer part numbers from a string
def extract_part_numbers(t: str) -> List[int]:
    return sorted({int(x) for x in re.findall(r"\d+", t)})

def update_session(models: list[str], parts: list[str]) -> None:
    """Push new codes into SESSION_STATE (dedup + keep latest first)."""
    if models:
        SESSION_STATE["models"] = [*dict.fromkeys(models + SESSION_STATE["models"])]
    if parts:
        SESSION_STATE["parts"]  = [*dict.fromkeys(parts  + SESSION_STATE["parts"])]


# ---- 1. function definition ----
def fallback_codes(models: list[str],
                   parts: list[int],
                   current_q: str) -> tuple[list[str], list[int]]:   #  ← add param
    lower_q = current_q.lower()
    used_models = models or (
        SESSION_STATE["models"][:1]
        if re.search(r"\bthis model|that model\b", lower_q) else []
    )
    used_parts = parts or (
        [int(SESSION_STATE["parts"][0])]
        if re.search(r"\bthis part|that part\b", lower_q) and SESSION_STATE["parts"] else []
    )
    return used_models, used_parts



# ───── Visible‑text helper ─────
def visible_text_from_html(html: str) -> str:
    soup_full = BeautifulSoup(html, "html.parser")
    main = soup_full.select_one("body > main") or soup_full
    soup = BeautifulSoup(str(main), "html.parser")
    for box in soup.select("div.mb-4[data-collapse-container]"):
        hdr = box.find("div", class_="section-title")
        # If the section title matches "Customer Reviews" or "Customer Repair Stories" (case-insensitive),
        # remove the entire section from the HTML to exclude user-submitted content.
        if hdr and re.match(r"Customer\s+(Reviews|Repair Stories)$", hdr.get_text(strip=True), re.I):
            box.decompose()
    qa = soup.select_one("#QuestionsAndAnswersContent")
    if qa:
        qa.decompose()
    return soup.get_text(" ", strip=True)

# ───── Selenium link discovery (uses caches first) ─────
NO_MATCH = "we couldn't find a match for"

# Discover link and page type (MODEL or PART) via Selenium scraping with caching
def discover_link(token: str) -> Tuple[str, str | None, str | None]:
    # ---------- cache hit ----------
    if token in MODEL_HTML_CACHE:
        d = MODEL_HTML_CACHE[token]
        return "MODEL", d["url"], d["html"]
    if token in PART_HTML_CACHE:
        d = PART_HTML_CACHE[token]
        return "PART", d["url"], d["html"]

    opts = FirefoxOptions()
    opts.add_argument("--headless")
    drv = webdriver.Firefox(options=opts)
    wait = WebDriverWait(drv, 15)
    try:
        if any(c.isalpha() for c in token):
            u = f"https://www.partselect.com/Models/{token}/"
            h = fetch_html(u)
            if h and token.lower() in h.lower() and NO_MATCH not in h.lower():
                MODEL_HTML_CACHE[token] = {"url": u, "html": h}
                _write_cache(MODEL_PAGE_FILE, MODEL_HTML_CACHE)
                return "MODEL", u, h

        drv.get("https://www.partselect.com/")
        box = wait.until(EC.presence_of_element_located((By.ID, "searchboxInput")))
        box.send_keys(token, Keys.ENTER)
        wait.until(lambda d: d.current_url != "https://www.partselect.com/")
        cur = drv.current_url
        if "/Models/" in cur or re.search(r'/PS\d+', cur):
            h = fetch_html(cur)
            if NO_MATCH in h.lower():
                return "NONE", None, None
            cls = "MODEL" if "/Models/" in cur else "PART"
            cache = MODEL_HTML_CACHE if cls == "MODEL" else PART_HTML_CACHE
            cache[token] = {"url": cur, "html": h}
            _write_cache(MODEL_PAGE_FILE if cls == "MODEL" else PART_PAGE_FILE, cache)
            return cls, cur, h

        for a in drv.find_elements(By.XPATH, '//main//a[@href]'):
            href = a.get_attribute("href") or ""
            if token.lower() not in href.lower():
                continue
            drv.get(href)
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            h = fetch_html(href)
            if NO_MATCH in h.lower():
                return "NONE", None, None
            cls = "MODEL" if "/Models/" in href else "PART"
            cache = MODEL_HTML_CACHE if cls == "MODEL" else PART_HTML_CACHE
            cache[token] = {"url": href, "html": h}
            _write_cache(MODEL_PAGE_FILE if cls == "MODEL" else PART_PAGE_FILE, cache)
            return cls, href, h

        return "NONE", None, None
    finally:
        drv.quit()

# ───── Compatible‑part scraping (unchanged logic) ─────
NEXT_RE = re.compile(r'/Parts/\?start=')

# Extract manufacturer and part codes from a model page
def _extract_codes(html: str) -> Set[str]:
    soup = BeautifulSoup(html, "html.parser")
    out: Set[str] = set()
    for span in soup.select("span.bold"):
        if (span.get_text(strip=True) in {"PartSelect #:", "Manufacturer #:"}
                and (txt := (span.next_sibling or "").strip())):
            out.add(txt)
    return out

# Get all part numbers associated with a given model
def get_model_parts(model_code: str) -> List[str]:
    url = f"https://www.partselect.com/Models/{model_code}/Parts/"
    found: Set[str] = set()
    while url:
        html = fetch_html(url)
        if not html:
            break
        found |= _extract_codes(html)
        soup = BeautifulSoup(html, "html.parser")
        nxt = soup.find("a", href=NEXT_RE, string=lambda s: s and "Next" in s)
        url = None
        if nxt and nxt.get("href"):
            href = nxt["href"]
            url = href if href.startswith("http") else f"https://www.partselect.com{href}"
    return sorted(found)

# ───── Prompt builder ─────
def build_prompt(question: str, snippets: Dict[str, List[str]], compat: bool) -> str:
    new_blob = "\n\n".join(
        f"Visible text for {t}:\n{'\n'.join(snippets[t])}" for t in snippets
    )
    BLOB_HISTORY.insert(0, new_blob)
    BLOB_HISTORY[:] = BLOB_HISTORY[:HISTORY_LIMIT]

    compat_lines = "\n".join(f"{m} compatible_parts: {MODEL_COMPAT[m]}" for m in MODEL_COMPAT) if compat else ""
    qa_block = "\n".join(
        f"PREVIOUS QUESTION: {q}\nPREVIOUS ANSWER: {a}" for q, a in reversed(QA_HISTORY[-QA_LIMIT:])
    ) or "None"


    prior = []
    valid_models = set(MODEL_DICT.keys())
    valid_parts  = set(PART_DICT.keys())

    # All tokens detected in the question
    all_models = set(extract_model_numbers(question))
    all_parts  = set(map(str, extract_part_numbers(question)))

    invalid_models = sorted(all_models - valid_models)
    invalid_parts  = sorted(all_parts - valid_parts)

    if valid_models and valid_parts:
        prior.append(f"Detected valid model codes: {sorted(valid_models)}")
        prior.append(f"Detected valid part numbers: {sorted(valid_parts)}")
    elif valid_models and not valid_parts:
        prior.append(f"Detected valid model codes: {sorted(valid_models)}")
    elif valid_parts and not valid_models:
        prior.append(f"Detected valid part numbers: {sorted(valid_parts)}")
    else:
        all_invalid = sorted(all_models | all_parts)
        invalid_codes = ", ".join(all_invalid) if all_invalid else "NONE"
        prior.append(f"No valid model or part pages were found for the detected code(s): {invalid_codes}")

    blob = BLOB_HISTORY[0] if BLOB_HISTORY else ""
    return textwrap.dedent(f"""
        You are a helpful assistant.
        Answer ONLY the CURRENT QUESTION; everything else is background.

        ==================  CURRENT QUESTION  =============
        {question}
        ===================================================

        ==================  BACKGROUND  ==================
        {' | '.join(prior)}
        {compat_lines}

        Previous interactions:
        {qa_block}

        Visible‑text snippets:
        {blob}
    """).strip()

# ───── Firecrawl research agent (persist & expose visited links) ─────
def research_agent(
    question: str,
    max_links: int = 300,
    num_best_links: int = 2,          # how many links to follow per page
) -> str:
    """
    Link‑guided browser‑agent that explores partselect.com when the user’s
    query lacks explicit model/part codes.

    NEW v2:
        • Persists every successful URL into visited_links.json
        • Adds those persisted links to the link‑selection menu so the LLM
          can choose from BOTH freshly‑scraped links and prior‑session links.
        • Allows the LLM to choose up to `num_best_links` links at each depth.
        • **Homepage (https://www.partselect.com/) is no longer included**
          in the final visited‑link chain or persisted history.
    """
    global VISITED_LINKS

    try:
        logger.info("[AGENT] No codes detected – invoking Firecrawl research agent")

        start_url = "https://www.partselect.com/"

        # Breadth‑first queue of (url, depth)
        queue: List[Tuple[str, int]] = [(start_url, 0)]
        visited: Set[str]            = set()
        visited_urls: List[str]      = []          # ordered chain of pages actually visited (ex‑home)
        page_snippets: List[str]     = []
        MAX_DEPTH                    = 2

        while queue:
            current_url, depth = queue.pop(0)

            if current_url in visited:
                continue
            visited.add(current_url)

            # Skip recording the homepage in final chain / persistence
            is_homepage = current_url.rstrip("/") == start_url.rstrip("/")
            if not is_homepage:
                visited_urls.append(current_url)

            # ---------- persist globally visited links (skip homepage) ----------
            if not is_homepage and current_url not in VISITED_LINKS:
                VISITED_LINKS.append(current_url)
                _write_cache(VISITED_LINKS_FILE, VISITED_LINKS)

            html = fetch_html(current_url)
            if not html:
                logger.error(f"[AGENT ERROR] Empty HTML for {current_url}")
                continue
            logger.info(f"[AGENT] [Depth {depth}] Scraped {current_url} ({len(html)} chars)")

            # take ~4 kB of visible text (homepage text still useful for nav)
            page_snippets.append(
                f"Page {depth} ({current_url}):\n{visible_text_from_html(html)[:4000]}"
            )

            # Stop expanding if we've hit the depth limit
            if depth >= MAX_DEPTH:
                continue

            # -------- collect links from current page --------
            page_links: List[Tuple[str, str]] = []  # (text, url)
            soup = BeautifulSoup(html, "html.parser")
            for a in soup.select("a[href]"):
                if len(page_links) >= max_links:
                    break
                href = a.get("href") or ""
                if href.startswith("#"):
                    continue
                full = href if href.startswith("http") else f"https://www.partselect.com{href}"
                text = a.get_text(" ", strip=True) or full
                page_links.append((text[:60], full))

            # -------- append previously‑visited links --------
            extra_links: List[Tuple[str, str]] = []
            for u in VISITED_LINKS:
                if u not in visited and len(extra_links) < max_links:
                    extra_links.append((f"[VISITED] {u[:60]}", u))

            all_links: List[Tuple[str, str]] = page_links + extra_links
            if not all_links:
                continue

            # -------- build link menu & ask LLM for up to N choices --------
            link_menu = "\n".join(
                f"{i}. {txt} -> {url}" for i, (txt, url) in enumerate(all_links)
            )
            nav_system = (
                "You are a navigation assistant on partselect.com.\n"
                f"Return **UP TO {num_best_links} link numbers** (separated by spaces, commas, "
                "or newlines) that will best help answer the question. "
                "If none look helpful, reply STOP."
            )
            nav_user = (
                f"USER QUESTION:\n{question}\n\n"
                f"LINKS:\n{link_menu}\n\n"
                "Which link number(s) should I click?"
            )
            choice = ask_llm(nav_system, nav_user, timeout=60)

            # Parse all integers the model returned (maintaining order / de‑duping)
            idxs: List[int] = []
            for tok in re.findall(r"\d+", choice or ""):
                i = int(tok)
                if i not in idxs:
                    idxs.append(i)
            idxs = idxs[: max(1, num_best_links)]  # cap to requested number

            if not idxs:
                logger.info("[AGENT] LLM chose STOP or returned no valid indices")
                continue

            # -------- enqueue chosen links --------
            for idx in idxs:
                if idx >= len(all_links):
                    logger.info(f"[AGENT] Index {idx} out of range – skipping")
                    continue
                chosen_text, next_url = all_links[idx]
                if next_url in visited:
                    continue
                logger.info(
                    f"[AGENT] [Depth {depth}] queued #{idx} '{chosen_text}' → {next_url}"
                )
                queue.append((next_url, depth + 1))

        # -------- build final answer prompt --------
        context = "\n\n".join(page_snippets[-3:])
        qa_block = (
            "\n".join(
                f"PREVIOUS QUESTION: {q}\nPREVIOUS ANSWER: {a}"
                for q, a in QA_HISTORY[-5:]
            )
            or "None"
        )
        visited_block = (
            "\n".join(f"{i+1}. {u}" for i, u in enumerate(visited_urls)) or "None"
        )

        answer_prompt = (
            "You are a helpful appliance‑repair assistant.\n"
            "Using ONLY the context below (plus previous interactions for background), "
            "answer the user's question clearly and concisely.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"VISITED LINKS THIS SESSION:\n{visited_block}\n\n"
            f"PREVIOUS INTERACTIONS (context only):\n{qa_block}\n\n"
            f"QUESTION:\n{question}"
        )

        logger.info(f"[AGENT FINAL PROMPT]\n{answer_prompt}\n[AGENT FINAL PROMPT END]")
        logger.info(f"[AGENT] Final visited chain (ex‑home): {visited_urls}")

        answer = ask_llm("You are a helpful assistant.", answer_prompt, timeout=500)
        return answer or "Sorry, I couldn't find enough information to answer that."

    except Exception as e:
        logger.exception(f"[AGENT ERROR] {e}")
        return "Sorry, I ran into a problem while researching that."


# ---------- fallback search agent (unchanged except robust attr access) ----------
def fallback_search_agent(question: str) -> str:
    try:
        search_resp = fc_client.search(query=f"{question} site:partselect.com", limit=5)
        hits = (
            getattr(search_resp, "items", None)
            or getattr(search_resp, "results", None)
            or (search_resp.get("items") if isinstance(search_resp, dict) else None)
            or []
        )
    except Exception as e:
        logger.error(f"[AGENT ERROR] Firecrawl search failed: {e}")
        return ask_llm("You are a helpful appliance‑repair assistant.", question)

    snippets = []
    for hit in hits:
        url = getattr(hit, "url", None) or hit.get("url")
        if not url:
            continue
        html = fetch_html(url)
        if not html:
            continue
        text = visible_text_from_html(html)
        if any(k in text.lower() for k in re.findall(r"[A-Za-z]+", question.lower())):
            snippets.append(f"From {url}:\n{text[:1500]}")
        if len(snippets) >= 5:
            break

    if not snippets:
        return ask_llm("You are a helpful appliance‑repair assistant.", question)

    blob = "\n\n".join(snippets)
    prompt = textwrap.dedent(f"""
        You are a helpful appliance‑repair assistant.
        Use ONLY the information in the snippets below to answer the user's question.

        ----------------  SNIPPETS  ----------------
        {blob}

        ---------------  QUESTION  -----------------
        {question}
    """).strip()

    return ask_llm("You are a helpful appliance‑repair assistant.", prompt)

# ───── Question processor ─────
def process_question(current_q: str) -> str:
    print(f"[DEBUG] incoming question: {current_q!r}")

    compat = llm_is_compat(current_q)
    logger.info(f"[DEBUG] compat: {compat}")

    # ---------- raw code extraction ----------
    raw_models = extract_model_numbers(current_q)
    raw_parts  = extract_part_numbers(current_q)
    # remove digits that belong to a model string
    digits     = {int(d) for m in raw_models for d in re.findall(r"\d+", m)}
    raw_parts  = [p for p in raw_parts if p not in digits]

    # ---------- fallback to conversation state ----------
    
    # ---- 2. call site inside process_question ----
    models, parts = fallback_codes(raw_models, raw_parts, current_q)  #  ← pass arg
    print(f"[DEBUG] models after fallback: {models}, parts after fallback: {parts}")

    # keep state fresh for the *next* turn
    update_session(models, [str(p) for p in parts])

    tokens = models + [str(p) for p in parts]

    # ───── NEW: only run research_agent when *no* codes are present ─────
    if not tokens:
        return research_agent(current_q)              

    snippets: Dict[str, List[str]] = {}
    # ---------- cached scraping ----------
    for tok in tokens:
        cls, url, html = discover_link(tok)
        if cls == "NONE":
            continue
        if cls == "MODEL":
            MODEL_DICT.setdefault(tok, url)
        elif cls == "PART":
            PART_DICT.setdefault(tok, url)
        if html:
            snippets[tok] = [visible_text_from_html(html)]

        # ---------- persist discovered links ----------
        if url and url not in VISITED_LINKS:
            VISITED_LINKS.append(url)
            _write_cache(VISITED_LINKS_FILE, VISITED_LINKS)
    

    # ---------- compatibility cache ----------
    if compat == "YES":
        for m in MODEL_DICT:
            if m not in MODEL_COMPAT:
                MODEL_COMPAT[m] = get_model_parts(m)
                _write_cache(COMPAT_CACHE_FILE, MODEL_COMPAT)

    # ---------- summarise & answer ----------
    prompt = build_prompt(current_q, snippets, compat == "YES")
    logger.info(f"[DEBUG] prompt: {prompt}")
    return ask_llm("You are a helpful assistant.", prompt)

# ───── FastAPI setup ─────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(req: ChatRequest):
    start = time.time()
    ans = process_question(req.message)
    QA_HISTORY.append((req.message, ans))
    QA_HISTORY[:] = QA_HISTORY[-QA_LIMIT:]
    logger.info(f"[DEBUG] handled in {int(time.time() - start)}s")
    return {"reply": ans}
