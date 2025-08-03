# ──────────────────────────── FLOW GRAPH ─────────────────────────────
#
#                          ┌──────────────────────┐
#                          │     User Prompt      │
#                          └─────────┬────────────┘
#                                    │
#                                    ▼
#                            analyse(prompt:str)
#                                    │
#                                    ▼
#                 +────────── Check for compat intent ──────────+
#                 │                                             │
#                 ▼                                             ▼
#     If compat → prewarm MODEL_COMPAT            else → skip prewarm
#                 │                                             │
#                 ▼                                             ▼
#                          ┌──────────────────────┐
#                          │  Orchestration Loop  │  ← max 6 steps
#                          └─────────┬────────────┘
#                                    ▼
#                           call_planner(history) ─────┐
#                                    │                 │
#                                    ▼                 │
#                          ┌──────────────────────┐    │
#                          │ JSON tool decision   │    │
#                          └─────────┬────────────┘    │
#                                    ▼                 │
#                  ┌──────────────────────────────────────────────┐
#                  │ TOOL_REGISTRY[action]                        │
#                  │    scrape_model(model_id)                    │
#                  │    scrape_part(part_id)                      │
#                  │    get_compat(model_id|part_id)              │
#                  │    firecrawl_research(query)                 │
#                  │    fill_context(question)                    │
#                  └────────────────────┬─────────────────────────┘
#                                       ▼
#                             Execute selected tool
#                                       │
#                                       ▼
#                  ┌────────────────────────────────────┐
#                  │ Log + inject TOOL_RESULT into chat │
#                  └────────────────────┬───────────────┘
#                                       ▼
#                           Back to call_planner(...) ←──────┐
#                                       │                    │
#                          if "FINAL_ANSWER"                ┌▼─────────────┐
#                          with non-empty text ────────────►│   Return     │
#                                                           │ Final Answer │
#                                                           └──────────────┘
#
# ──────────────────────────────────────────────────────────────────────
#  Highlights:
#   • Session state tracks models/parts for pronoun resolution
#   • Compatibility flow handles both model→parts and part→model
#   • Firecrawl fetches pages first; Selenium fallback adds robustness
#   • Research agent handles ambiguous or unknown questions with depth-limited BFS
#   • Planner loop chooses tool, injects result, re-calls planner with context
#


import logging, os, re, textwrap, time, json
from enum import Enum
from typing import Literal, Tuple, List, Dict, Set, Union

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

import threading, time, logging
from selenium import webdriver
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.common.exceptions import TimeoutException, WebDriverException
# ─────────────────────────  GLOBAL CONFIG  ──────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn.error")

load_dotenv("python-dotenv.env")

DEEPSEEK_API_KEY = os.getenv("DEEP_SEEK_WEB_API_KEY")
if not DEEPSEEK_API_KEY:
    logger.error("DeepSeek API key not found in DEEP_SEEK_WEB_API_KEY")

client   = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

FC_KEY   = os.getenv("FIRECRAWL_API_KEY")
if not FC_KEY:
    raise RuntimeError("FIRECRAWL_API_KEY missing in environment")
fc_client = FirecrawlApp(api_key=FC_KEY)

RATE_LIMIT_SLEEP, MAX_RETRIES = 60, 3           # Firecrawl retry params
HISTORY_LIMIT = QA_LIMIT = 20                   # prompt / answer history caps

# ─────────────────────  DISK‑PERSISTENT CACHES  ─────────────────────
COMPAT_CACHE_FILE  = "model_compat_cache.json"
MODEL_PAGE_FILE    = "model_page_cache.json"
PART_PAGE_FILE     = "part_page_cache.json"
VISITED_LINKS_FILE = "visited_links.json"

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

MODEL_COMPAT      = _load_cache(COMPAT_CACHE_FILE, {})
MODEL_HTML_CACHE  = _load_cache(MODEL_PAGE_FILE, {})
PART_HTML_CACHE   = _load_cache(PART_PAGE_FILE, {})
VISITED_LINKS     = _load_cache(VISITED_LINKS_FILE, [])

# ──────────────────────  IN‑MEMORY SESSION STATE  ────────────────────
BLOB_HISTORY: List[str]            = []
QA_HISTORY:   List[Tuple[str,str]] = []
MODEL_DICT:   Dict[str,str]        = {}
PART_DICT:    Dict[str,str]        = {}
SESSION_STATE = {"models": [], "parts": []}     # for pronoun references

# ─────────────────────────  LLM HELPER  ──────────────────────────────
def ask_llm(system: str, user: Union[str, List[Dict]], timeout: int = 500) -> str:
    """
    Generic DeepSeek wrapper. Accepts either a user string or
    an already‑constructed messages list.
    """
    if isinstance(user, str):
        msgs = [{"role": "system", "content": system},
                {"role": "user",   "content": user}]
    else:
        msgs = [{"role": "system", "content": system}] + user

    try:
        # Log the full prompt for debugging
        logger.info("[DEBUG] Full planner prompt:\n" + json.dumps(msgs, indent=2))

        r = client.chat.completions.create(
            model="deepseek-chat",
            messages=msgs,
            timeout=timeout,
            stream=False,
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"[LLM WARNING] {e}")
        return ""

# ──────────────────────  REGEX ANALYSIS HELPERS  ─────────────────────
COMPAT_RE     = re.compile(r"\b(compatib(?:le|ility)|fit|works?\s+with|match(?:es)?)\b", re.I)
MODEL_PATTERN = re.compile(r"\b([A-Za-z0-9][A-Za-z0-9\-\.]{5,})\b")
BLOCKLIST     = {"COMPATIBLE", "MODEL", "PART", "NUMBER"}

def llm_is_compat(q: str) -> Literal["YES", "NO"]:
    if COMPAT_RE.search(q):
        return "YES"
    return "YES" if ask_llm("Return ONLY YES if asking part‑model compatibility, else NO.", q, 120) == "YES" else "NO"

def extract_model_numbers(t: str) -> List[str]:
    seen, out = set(), []
    for raw in MODEL_PATTERN.findall(t):
        m = raw.rstrip("?.!,")
        if m.upper() in BLOCKLIST or m.isalpha(): continue
        if m not in seen:
            seen.add(m); out.append(m)
    return out

# ───────── Part‑ID extraction ─────────────────────────
def extract_part_numbers(text: str) -> List[str]:
    """
    Return every PartSelect‑style ID found in *text*.

    ▸ Accepts either pure‑numeric codes (“279570”) **or**
      the alphanumeric PS‑prefix form (“PS373087”).

    ▸ Everything is normalised to UPPER‑CASE **strings**.

    Example:
        "I need PS11752778 or 279570" ➜ ["279570", "PS11752778"]
    """
    #  (?:PS)?   – optional 'PS' / 'ps'
    #  \d{3,}    – three or more digits
    tokens = re.findall(r"(?:PS)?\d{3,}", text, flags=re.I)
    return sorted({tok.upper() for tok in tokens})


# ───────── Conversation‑state helper ─────────────────
def update_session(models: list[str], parts: list[str]) -> None:
    """
    Keep a rolling list (most‑recent first) of the model and part codes
    mentioned so far in this session.  *All* codes are stored as strings.
    """
    if models:
        SESSION_STATE["models"] = [*dict.fromkeys(models + SESSION_STATE["models"])]

    if parts:
        SESSION_STATE["parts"]  = [*dict.fromkeys(parts  + SESSION_STATE["parts"])]


# ───────── Pronoun fallback helper ───────────────────
# ───────────────────  REGEX helper: pronoun fallback  ───────────────────
def fallback_codes(models: list[str],
                   parts:  list[str],   # ← keep parts as strings
                   q: str) -> tuple[list[str], list[str]]:
    """
    If the user says “this model / that part”, substitute the last‑seen
    codes stored in SESSION_STATE.  **Part numbers stay as strings** so we
    don’t trip over non‑numeric IDs like “PS11752778”.
    """
    lower = q.lower()

    if not models and re.search(r"\b(this|that)\s+model\b", lower):
        models = SESSION_STATE["models"][:1]          # most‑recent model

    if not parts and re.search(r"\b(this|that)\s+part\b", lower):
        parts = SESSION_STATE["parts"][:1]            # most‑recent part

    return models, parts



# ────────────────────────  SCRAPE UTILITIES  ────────────────────────
def visible_text_from_html(html: str) -> str:
    """Return main visible text, remove reviews/QA sections."""
    soup_full = BeautifulSoup(html, "html.parser")
    main      = soup_full.select_one("body > main") or soup_full
    soup      = BeautifulSoup(str(main), "html.parser")
    for box in soup.select("div.mb-4[data-collapse-container]"):
        hdr = box.find("div", class_="section-title")
        if hdr and re.match(r"Customer\s+(Reviews|Repair Stories)$", hdr.get_text(strip=True), re.I):
            box.decompose()
    qa = soup.select_one("#QuestionsAndAnswersContent")
    if qa: qa.decompose()
    return soup.get_text(" ", strip=True)

# ─────────────────────  FIRECRAWL fetch (single‑threaded)  ──────────────


FIRECRAWL_LOCK = threading.Lock()
MAX_RETRIES = 3
RATE_LIMIT_SLEEP = 2  # seconds
logger = logging.getLogger(__name__)

def fetch_html(url: str) -> str:
    """
    Fetch *url* using Firecrawl first, falling back to Selenium if needed.
    A global mutex (`FIRECRAWL_LOCK`) ensures only one scrape at a time.
    """
    with FIRECRAWL_LOCK:
        # ───── Phase 1: Try Firecrawl first ─────
        for attempt in range(1, MAX_RETRIES + 1):
            logger.info(f"[FIRECRAWL] Scraping {url} (try {attempt}/{MAX_RETRIES})")
            try:
                r = fc_client.scrape_url(url=url, formats=["html"])
                if r.html:
                    return r.html
                logger.warning("[FIRECRAWL] Empty HTML returned")
            except Exception as e:
                transient = any(tok in str(e) for tok in ("429", "Timeout"))
                if transient and attempt < MAX_RETRIES:
                    logger.warning(f"[FIRECRAWL WAIT] transient error: {e}. Sleeping {RATE_LIMIT_SLEEP}s.")
                    time.sleep(RATE_LIMIT_SLEEP)
                    continue
                logger.error(f"[FIRECRAWL ERROR] {e}")
                break  # Fail fast to fallback

        logger.warning("[FIRECRAWL] All attempts failed — falling back to Selenium")

        # ───── Phase 2: Fallback to Selenium ─────
        options = FirefoxOptions()
        options.headless = True

        for attempt in range(1, MAX_RETRIES + 1):
            logger.info(f"[SELENIUM] Scraping {url} (try {attempt}/{MAX_RETRIES})")
            driver = None
            try:
                driver = webdriver.Firefox(options=options)
                driver.set_page_load_timeout(20)
                driver.get(url)
                html = driver.page_source
                return html
            except (TimeoutException, WebDriverException) as e:
                logger.warning(f"[SELENIUM ERROR] transient error: {e}")
                if attempt < MAX_RETRIES:
                    logger.warning(f"[SELENIUM WAIT] Sleeping {RATE_LIMIT_SLEEP}s before retry.")
                    time.sleep(RATE_LIMIT_SLEEP)
                else:
                    logger.error(f"[SELENIUM ERROR] failed after {MAX_RETRIES} retries: {e}")
                    return ""
            finally:
                if driver:
                    try:
                        driver.quit()
                    except Exception:
                        logger.debug("[SELENIUM CLEANUP] Failed to quit driver")

    return ""



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
# ─── Link‑based code‑type classifier ──────────────────────────────────
def lookup_code_type(tok: str) -> Literal["MODEL", "PART", "UNKNOWN"]:
    """
    Decide whether *tok* is a model number or a part number by
    consulting the HTML caches first and, if necessary, by calling
    discover_link(tok) (which does a quick site lookup and fills the cache).
    """
    if tok in MODEL_HTML_CACHE:
        return "MODEL"
    if tok in PART_HTML_CACHE:
        return "PART"

    cls, _, _ = discover_link(tok)          # may be "MODEL", "PART", or "NONE"
    if cls in {"MODEL", "PART"}:
        return cls
    return "UNKNOWN"

# compatible‑parts fetch
NEXT_RE = re.compile(r'/Parts/\?start=')
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

# ────────────────────────  TOOL DEFINITIONS  ────────────────────────
class Action(Enum):
    SCRAPE_MODEL   = "scrape_model"
    SCRAPE_PART    = "scrape_part"
    GET_COMPAT     = "get_compat"
    RESEARCH       = "firecrawl_research"
    FILL_CONTEXT   = "fill_context"
    FINAL_ANSWER   = "final_answer"

# ▼▼▼  REPLACE from here … ▼▼▼
def _retry_fetch_visible(url: str, wait_sec: int = 2) -> str:
    """
    Open the URL in a headless browser, wait a couple of seconds, and
    extract the visible text again.  Used when the first Firecrawl scrape
    returns almost‑empty HTML (JS‑rendered pages, slow servers, etc.).
    """
    opts = FirefoxOptions()
    opts.add_argument("--headless")
    drv = webdriver.Firefox(options=opts)
    try:
        drv.get(url)
        time.sleep(wait_sec)                      # give JS a moment
        html = drv.page_source or ""
        return visible_text_from_html(html)
    finally:
        drv.quit()


def scrape_model(model_id: str) -> dict:
    cls, url, html = discover_link(model_id)
    if cls != "MODEL":
        return {"error": "not_found"}

    text = visible_text_from_html(html)

    # ── First‑level retry: live browser fetch ─────────────────────────
    if not text.strip():
        logger.info(f"[SCRAPE_MODEL] Empty text – live‑retry on {url}")
        text = _retry_fetch_visible(url)

    # ── Second‑level fallback: research_agent summary ─────────────────
    if not text.strip():
        logger.warning(f"[SCRAPE_MODEL] Still empty – using research_agent fallback")
        text = research_agent(f"Summarise useful information from {url}")

    MODEL_DICT.setdefault(model_id, url)
    return {"url": url, "text": text}


def scrape_part(part_id: str) -> dict:
    cls, url, html = discover_link(str(part_id))
    if cls != "PART":
        return {"error": "not_found"}

    text = visible_text_from_html(html)

    if not text.strip():
        logger.info(f"[SCRAPE_PART] Empty text – live‑retry on {url}")
        text = _retry_fetch_visible(url)

    if not text.strip():
        logger.warning(f"[SCRAPE_PART] Still empty – using research_agent fallback")
        text = research_agent(f"Summarise useful information from {url}")

    PART_DICT.setdefault(str(part_id), url)
    return {"url": url, "text": text}
# ▲▲▲  … to here  ▲▲▲


# ▼▼▼  REPLACE the whole get_compat() implementation ▼▼▼
def get_compat(code: str) -> dict:
    """
    Unified compatibility checker.

    ── If *code* is a **MODEL**:
           • Ensure MODEL_COMPAT cache is populated (get_model_parts).
           • Return {"type": "model", "compatible_parts": [...]}

    ── If *code* is a **PART**:
           • Scrape the part page and extract its MODEL cross-reference list.
           • For *every* model currently stored in SESSION_STATE["models"],
             populate / refresh MODEL_COMPAT.
           • Build a mapping {"<model>": True/False} indicating whether that
             part appears in each model’s parts list.
           • Return
                 {
                   "type": "part",
                   "part":  code,
                   "crossref_models": [...],      # from part page
                   "session_models":   {...}      # True/False per model
                 }

    ── Otherwise returns {"error": "not_resolved"}.
    """
    # ── Determine whether the identifier is a model or a part ─────────
    kind, _, _ = discover_link(code)           # "MODEL", "PART", or "NONE"
    kind = (kind or "NONE").upper()

    # ────────────────── MODEL branch ──────────────────────────────────
    if kind == "MODEL":
        if code not in MODEL_COMPAT:
            logger.info(f"[COMPAT CACHE] populating parts for {code}")
            MODEL_COMPAT[code] = get_model_parts(code)
            _write_cache(COMPAT_CACHE_FILE, MODEL_COMPAT)
        return {
            "type": "model",
            "model": code,
            "compatible_parts": MODEL_COMPAT[code],
        }

    # ────────────────── PART branch ───────────────────────────────────
    if kind == "PART":
        # scrape_part returns visible text already (will retry/fallback)
        part_page = scrape_part(code)
        xmodels   = extract_crossref_models(part_page.get("text", ""))

        # make sure every session model has a parts table
        # AFTER ─ also refresh if the cached list is empty
        for mdl in SESSION_STATE["models"]:
            if mdl not in MODEL_COMPAT or not MODEL_COMPAT[mdl]:      # ← same trick
                logger.info(f"[COMPAT CACHE] populating parts for {mdl}")
                MODEL_COMPAT[mdl] = get_model_parts(mdl)
                _write_cache(COMPAT_CACHE_FILE, MODEL_COMPAT)


        # build comparison dict {model: True/False}
        comparison = {
            mdl: code in MODEL_COMPAT.get(mdl, [])
            for mdl in SESSION_STATE["models"]
        }

        return {
            "type": "part",
            "part": code,
            "crossref_models": xmodels,
            "session_models": comparison,
        }

    # ────────────────── Unknown branch ────────────────────────────────
    logger.warning(f"[get_compat] '{code}' could not be resolved as model or part")
    return {"error": "not_resolved"}
# ▲▲▲  END of get_compat() replacement  ▲▲▲


def extract_crossref_models(text: str) -> list[str]:
    matches = re.findall(r"Kenmore\s+\d{11}", text)
    return sorted(set(matches))  # unique model IDs

def firecrawl_research(query: str) -> dict:
    return {"answer": research_agent(query)}

def fill_context(question: str) -> dict:
    ctx = ask_llm("Briefly infer missing appliance context for this question:", question)
    return {"context": ctx}

TOOL_REGISTRY = {
    Action.SCRAPE_MODEL:  scrape_model,
    Action.SCRAPE_PART:   scrape_part,
    Action.GET_COMPAT:    get_compat,
    Action.RESEARCH:      firecrawl_research,
    Action.FILL_CONTEXT:  fill_context,
}

# ───────────────────  PLANNER PROMPT & CALLER  ──────────────────────
PLANNER_SYS = """
You are a TOOL‑PLANNING agent. Available tools:
- scrape_model(model_id)
- scrape_part(part_id)
- get_compat(model_id)
- firecrawl_research(query)
- fill_context(question)

Decide the next action and return ONLY valid JSON:
{ "action": "<tool | FINAL_ANSWER>", "args": { ... } }

Rules:
- Do NOT return FINAL_ANSWER if the previous tool result is empty or lacks information (e.g., get_compat returned an empty list, or a scrape failed).
- If a part/model was not found or compatible_parts is empty, consider using firecrawl_research or scrape_model/scrape_part.
- Use fill_context if the user question seems ambiguous or lacks required identifiers.
- Always extract information from previous TOOL_RESULTs before deciding next steps.
- If the user is asking about compatibility, only mention specific models that were mentioned in the current or previous user inputs.
- Do not list all compatible models unless the user explicitly asks for all of them.
"""


def call_planner(msgs: List[Dict]) -> dict:
    """Ask planner LLM, strip ``` fences, return dict."""
    raw = ask_llm(PLANNER_SYS, msgs, 300).strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        plan = json.loads(raw)
    except json.JSONDecodeError:
        logger.error(f"[PLANNER] bad JSON after strip: {raw}")
        plan = {"action": "FINAL_ANSWER", "args": {"answer": "Planner error."}}
    logger.info(f"[PLANNER] {plan}")
    return plan

# ───────────────────────  ANALYSIS PRE‑PROCESS  ─────────────────────
def analyse(q: str) -> dict:
    compat_intent = llm_is_compat(q) == "YES"

    # raw numeric / alphanumeric chunks from the text
    tokens = extract_model_numbers(q)

    # ↓↓↓── REPLACE FROM HERE ──────────────────────────────────────────
    models, parts = [], []
    for tok in tokens:
        code_type = lookup_code_type(tok)
        if code_type == "MODEL":
            models.append(tok.upper())
        elif code_type == "PART":
            parts.append(tok.upper())

    # pronoun / last‑seen fall‑back
    models, parts = fallback_codes(models, parts, q)
    update_session(models, parts)          # parts are already strings

    # ↑↑↑── REPLACE UNTIL HERE ─────────────────────────────────────────

    return {"compat_intent": compat_intent,
            "models": models,
            "parts":  parts}


# ───────────────────────  ORCHESTRATOR LOOP  ────────────────────────
# ───────────────────────  ORCHESTRATOR LOOP  ────────────────────────
def process_question(q: str) -> str:
    logger.info(f"[DEBUG] incoming: {q!r}")

    analysis = analyse(q)

    # NEW – pre‑warm only when we have *no* compat list at all
    if analysis["compat_intent"]:
        for mdl in SESSION_STATE["models"]:
            if mdl not in MODEL_COMPAT or not MODEL_COMPAT[mdl]:   # ← check emptiness
                logger.info(f"[COMPAT CACHE] populating parts for {mdl}")
                MODEL_COMPAT[mdl] = get_model_parts(mdl)
                _write_cache(COMPAT_CACHE_FILE, MODEL_COMPAT)


    messages = [
        {"role": "user",   "content": q},
        {"role": "system", "content": f"analysis: {analysis}"},
    ]

    for step in range(6):  # max 6 tool calls
        plan = call_planner(messages)
        raw_action = plan["action"]
        logger.info(f"[DEBUG] planner raw action: {raw_action!r}")

        # ---------- normalise action ----------
        norm = re.sub(r"[\s\-]", "_", raw_action.strip().lower())
        alias_map = {
            "scrapemodel":        "scrape_model",
            "scrape_model":       "scrape_model",
            "scrapepart":         "scrape_part",
            "scrape_part":        "scrape_part",
            "getcompat":          "get_compat",
            "get_compat":         "get_compat",
            "firecrawlresearch":  "firecrawl_research",
            "firecrawl_research": "firecrawl_research",
            "fillcontext":        "fill_context",
            "fill_context":       "fill_context",
            "final_answer":       "final_answer",
        }
        norm = alias_map.get(norm, norm)

        # ---------- handle FINAL_ANSWER ----------
        if norm == "final_answer":
            answer = (plan.get("args", {}).get("answer") or
                      plan.get("args", {}).get("response", "")).strip()
            if not answer:
                logger.warning(f"[PLANNER] Empty FINAL_ANSWER – retrying planner (step {step})")
                continue
            logger.info(f"[DEBUG] FINAL ANSWER @step{step}: {answer}")
            return answer

        # ---------- resolve action enum and tool ----------
        try:
            action_enum = Action[norm.upper()]        # by Enum *name*
        except KeyError:
            matches = [a for a in Action if a.value == norm]  # by Enum value
            if not matches:
                logger.error(f"[PLANNER] unknown action after normalisation: {norm}")
                return f"Planner returned unknown action: {raw_action}"
            action_enum = matches[0]

        tool_fn = TOOL_REGISTRY[action_enum]

        # ---------- call tool and log result ----------
        if action_enum == Action.GET_COMPAT:
            code = plan["args"].get("model_id") or plan["args"].get("part_id") or list(plan["args"].values())[0]
            tool_out = tool_fn(code)
        else:
            tool_out = tool_fn(**plan["args"])


        # If we just asked for compat and got an empty list, push hint + retry
                # ---------- guard: get_compat must receive a MODEL id ----------
        # ▼▼▼  REPLACE this entire GET_COMPAT handling block ▼▼▼
        if action_enum == Action.GET_COMPAT:
            if tool_out.get("type") == "model":
                compatible_parts = tool_out.get("compatible_parts", [])
                if compatible_parts:
                    part_table = "\n".join(f"- {p}" for p in compatible_parts)
                    logger.info(f"[DEBUG] Injecting full compatibility table into context (showing first 25 in logs):\n" +
                                "\n".join(f"- {p}" for p in compatible_parts[:25]))
                    messages.append({
                        "role": "system",
                        "content": f"Known compatible parts for model {tool_out['model']}:\n{part_table}"
                    })

                else:
                    logger.warning("[DEBUG] Model compat list empty – nudging planner to research")
                    messages.append({
                        "role": "assistant",
                        "content": "TOOL_RESULT get_compat (model): empty – consider research"
                    })
                continue   # let planner decide next step

            if tool_out.get("type") == "part":
                # Build a concise yes/no summary for the models in this session
                summary = "\n".join(
                    f"- {mdl}: {'✅' if ok else '❌'}"
                    for mdl, ok in tool_out.get("session_models", {}).items()
                ) or "No models referenced in this session."

                logger.info(f"[DEBUG] Injecting part-to-model comparison:\n{summary}")
                messages.append({
                    "role": "system",
                    "content": (
                        f"Compatibility check for part {tool_out['part']} against models "
                        f"in this conversation:\n{summary}"
                    )
                })
                continue   # hand control back to planner
# ▲▲▲  END of GET_COMPAT handling replacement  ▲▲▲





        logger.info(f"[DEBUG] {action_enum.value} → {tool_out}")

        messages.append({
            "role": "assistant",
            "content": f"TOOL_RESULT {action_enum.value}: {json.dumps(tool_out)}"
        })

    return "Sorry, I couldn't resolve that after several attempts."




# ──────────────────────────  FASTAPI APP  ───────────────────────────
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
    t0 = time.time()
    ans = process_question(req.message)
    QA_HISTORY.append((req.message, ans)); QA_HISTORY[:] = QA_HISTORY[-QA_LIMIT:]
    logger.info(f"[DEBUG] handled in {int(time.time()-t0)}s")
    return {"reply": ans}
