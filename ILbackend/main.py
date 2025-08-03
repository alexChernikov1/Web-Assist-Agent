from __future__ import annotations

# ════════════════════════════════════════════════════════════════════════
#  main.py — Firecrawl‑powered FastAPI backend with LangGraph orchestration
#  Uses LangChain + LangGraph `create_react_agent` for tool planning
# ════════════════════════════════════════════════════════════════════════
#  Debug prints and logs preserved 1‑for‑1
#  Major changes:
#    1. LangChain/LangGraph tool wrappers + agent construction
#    2. process_question() routes queries through the agent
# ------------------------------------------------------------------------

import os, re, json, time, logging, threading, textwrap, requests
from enum import Enum
from typing import Optional, Literal, Tuple, List, Dict, Set, Union

# ───────────────────────── Third‑party libraries ─────────────────────────
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
from selenium.common.exceptions import TimeoutException, WebDriverException
from dotenv import load_dotenv
from firecrawl import FirecrawlApp
from openai import OpenAI

# ─────────────── LangChain + LangGraph agent imports ───────────────
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from langgraph.prebuilt import create_react_agent

logger = logging.getLogger(__name__)
# ─────────────────────────  GLOBAL CONFIG  ──────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn.error")

load_dotenv("python-dotenv.env")

DEEPSEEK_API_KEY = os.getenv("DEEP_SEEK_WEB_API_KEY")
if not DEEPSEEK_API_KEY:
    logger.error("DeepSeek API key not found in DEEP_SEEK_WEB_API_KEY")

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

FC_KEY = os.getenv("FIRECRAWL_API_KEY")
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
PART_COMPAT_FILE = "part_compat_cache.json"

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
PART_COMPAT       = _load_cache(PART_COMPAT_FILE, {})
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
    tokens = re.findall(r"(?:PS)?\d{3,}", text, flags=re.I)
    return sorted({tok.upper() for tok in tokens})

# ───────── Conversation‑state helper ─────────────────
def update_session(models: list[str], parts: list[str]) -> None:
    if models:
        SESSION_STATE["models"] = [*dict.fromkeys(models + SESSION_STATE["models"])]
    if parts:
        SESSION_STATE["parts"]  = [*dict.fromkeys(parts  + SESSION_STATE["parts"])]

# ───────── Pronoun fallback helper ───────────────────
def fallback_codes(models: list[str], parts: list[str], q: str) -> tuple[list[str], list[str]]:
    lower = q.lower()
    if not models and re.search(r"\b(this|that)\s+model\b", lower):
        models = SESSION_STATE["models"][:1]
    if not parts and re.search(r"\b(this|that)\s+part\b", lower):
        parts = SESSION_STATE["parts"][:1]
    return models, parts

# ────────────────────────  SCRAPE UTILITIES  ────────────────────────
def visible_text_from_html(html: str) -> str:
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

def fetch_html(url: str) -> str:
    with FIRECRAWL_LOCK:
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
                break

        logger.warning("[FIRECRAWL] All attempts failed — falling back to Selenium")

        options = FirefoxOptions(); options.headless = True
        for attempt in range(1, MAX_RETRIES + 1):
            logger.info(f"[SELENIUM] Scraping {url} (try {attempt}/{MAX_RETRIES})")
            driver = None
            try:
                driver = webdriver.Firefox(options=options)
                driver.set_page_load_timeout(20)
                driver.get(url)
                return driver.page_source
            except (TimeoutException, WebDriverException) as e:
                logger.warning(f"[SELENIUM ERROR] {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(RATE_LIMIT_SLEEP)
                else:
                    logger.error(f"[SELENIUM ERROR] failed after {MAX_RETRIES} retries: {e}")
            finally:
                if driver:
                    try: driver.quit()
                    except Exception: logger.debug("[SELENIUM CLEANUP] Failed to quit driver")

    return ""

# ───────────────────────  DISCOVERY / CACHING  ──────────────────────
NO_MATCH = "we couldn't find a match for"

def discover_link(token: str) -> Tuple[str, str | None, str | None]:
    if token in MODEL_HTML_CACHE:  return "MODEL", MODEL_HTML_CACHE[token]["url"], MODEL_HTML_CACHE[token]["html"]
    if token in PART_HTML_CACHE:   return "PART",  PART_HTML_CACHE[token]["url"],  PART_HTML_CACHE[token]["html"]

    opts = FirefoxOptions(); opts.add_argument("--headless")
    drv = webdriver.Firefox(options=opts); wait = WebDriverWait(drv, 15)
    try:
        if any(c.isalpha() for c in token):
            u = f"https://www.partselect.com/Models/{token}/"
            h = fetch_html(u)
            if h and token.lower() in h.lower() and NO_MATCH not in h.lower():
                MODEL_HTML_CACHE[token] = {"url": u, "html": h}; _write_cache(MODEL_PAGE_FILE, MODEL_HTML_CACHE)
                return "MODEL", u, h

        drv.get("https://www.partselect.com/")
        box = wait.until(EC.presence_of_element_located((By.ID, "searchboxInput"))); box.send_keys(token, Keys.ENTER)
        wait.until(lambda d: d.current_url != "https://www.partselect.com/")
        cur = drv.current_url
        if "/Models/" in cur or re.search(r'/PS\d+', cur):
            h = fetch_html(cur); cls = "MODEL" if "/Models/" in cur else "PART"
            if NO_MATCH in h.lower(): return "NONE", None, None
            cache = MODEL_HTML_CACHE if cls == "MODEL" else PART_HTML_CACHE
            cache[token] = {"url": cur, "html": h}; _write_cache(MODEL_PAGE_FILE if cls=="MODEL" else PART_PAGE_FILE, cache)
            return cls, cur, h

        for a in drv.find_elements(By.XPATH, '//main//a[@href]'):
            href = a.get_attribute("href") or ""
            if token.lower() not in href.lower(): continue
            drv.get(href); wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            h = fetch_html(href); cls = "MODEL" if "/Models/" in href else "PART"
            if NO_MATCH in h.lower(): return "NONE", None, None
            cache = MODEL_HTML_CACHE if cls == "MODEL" else PART_HTML_CACHE
            cache[token] = {"url": href, "html": h}; _write_cache(MODEL_PAGE_FILE if cls=="MODEL" else PART_PAGE_FILE, cache)
            return cls, href, h
        return "NONE", None, None
    finally:
        drv.quit()

def lookup_code_type(tok: str) -> Literal["MODEL", "PART", "UNKNOWN"]:
    if tok in MODEL_HTML_CACHE: return "MODEL"
    if tok in PART_HTML_CACHE:  return "PART"
    cls, _, _ = discover_link(tok)
    return cls if cls in {"MODEL","PART"} else "UNKNOWN"

NEXT_RE = re.compile(r'/Parts/\?start=')
def _extract_codes(html: str) -> Set[str]:
    soup = BeautifulSoup(html, "html.parser"); out:set[str] = set()
    for span in soup.select("span.bold"):
        if (span.get_text(strip=True) in {"PartSelect #:", "Manufacturer #:"}
                and (txt := (span.next_sibling or "").strip())):
            out.add(txt)
    return out

def get_model_parts(model_code: str) -> List[str]:
    url = f"https://www.partselect.com/Models/{model_code}/Parts/"; found:Set[str] = set()
    while url:
        html = fetch_html(url)
        if not html: break
        found |= _extract_codes(html)
        soup = BeautifulSoup(html, "html.parser")
        nxt = soup.find("a", href=NEXT_RE, string=lambda s: s and "Next" in s)
        url = None
        if nxt and nxt.get("href"):
            href = nxt["href"]; url = href if href.startswith("http") else f"https://www.partselect.com{href}"
    return sorted(found)

# ─────────────────────  FIRECRAWL research agent  ────────────────────
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

# all scrape_model / scrape_part / get_compat / ... functions remain exactly as before
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

# ──────────────────────────────  NEW HELPERS  ──────────────────────────────


# Standard desktop UA so PartSelect is less likely to block us
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}



# ─────────────  NEW URL-RESOLVER  (JSON API → Selenium) ─────────────
def _resolve_part_url(
    part_code: str,
    *,
    session: requests.Session,
) -> str | None:
    """
    1. Try the public JSON search API.
    2. If that fails, fall back to a *visible* Firefox search.
    """
    # ❶ JSON search API
    try:
        api = f"https://www.partselect.com/api/search/?searchterm={part_code}"
        r   = session.get(api, headers=_HEADERS, timeout=12)
        r.raise_for_status()
        for bucket in (
            r.json().get("ExactPartPageResults", [])
            + r.json().get("PartPageResults", [])
            + r.json().get("PartResults", [])
        ):
            if str(bucket.get("PartNumber", "")).upper() == part_code.upper():
                return "https://www.partselect.com" + bucket["Url"]
    except Exception as e:
        logger.info(f"[resolve_part_url] search API failed: {e}")

    # ❷ Selenium fallback (visible Firefox)
    try:
        opts = FirefoxOptions()
        # comment‐out the next line so the browser window *is shown*
        # opts.add_argument("--headless")

        drv  = webdriver.Firefox(options=opts)
        drv.set_page_load_timeout(30)
        drv.get("https://www.partselect.com/")

        box = WebDriverWait(drv, 15).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "input.js-headerNavSearch")
            )
        )
        box.send_keys(part_code, Keys.ENTER)

        # wait until the URL actually switches to the part page
        WebDriverWait(drv, 15).until(
            lambda d: part_code.lower() in d.current_url.lower()
        )
        return drv.current_url

    except Exception as e:
        logger.warning(f"[resolve_part_url] selenium search failed: {e}")

    finally:
        try:
            drv.quit()
        except Exception:
            pass

    return None




logger = logging.getLogger(__name__)
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

def _scroll_collect_models(
    drv: webdriver.Firefox,
    container,               # WebElement for the inner scroll box
    *,
    pause: float = 1.0,
    max_idle_runs: int = 3,
) -> Set[str]:
    """
    Scroll the *container* element to its bottom over and over until
    no new model numbers appear for `max_idle_runs` consecutive scrolls.
    """
    models: Set[str] = set()
    idle_runs       = 0
    last_count      = -1

    def harvest():
        # only look **inside the container**
        for a in container.find_elements(By.CSS_SELECTOR, 'a[href^="/Models/"]'):
            t = a.text.strip()
            if t.isdigit():
                models.add(t)

    while idle_runs < max_idle_runs:
        harvest()
        if len(models) == last_count:
            idle_runs += 1          # nothing new → one idle cycle
        else:
            idle_runs = 0
        last_count = len(models)

        # scroll the inner box, not the window
        drv.execute_script(
            "arguments[0].scrollTop = arguments[0].scrollHeight;", container
        )
        time.sleep(pause)

    harvest()                       # final sweep
    return models


# ──────────────────────────── MAIN SCRAPER ────────────────────────────
def get_part_models(
    part_code: str,
    *,
    session: Optional[requests.Session] = None,
    pause: float = 1.0,
) -> List[str]:
    """
    **Scroll-only** scraper that targets the *nested* scrollable list of
    compatible models.  Opens a **visible** Firefox window so you can
    watch each scroll step.
    """
    session = session or requests.Session()
    url     = _resolve_part_url(part_code, session=session)
    if not url:
        logger.warning(f"[get_part_models] couldn’t resolve URL for {part_code}")
        return []

    logger.info(f"[get_part_models] scraping {url}")

    opts = FirefoxOptions()               # visible browser (no --headless flag)
    drv  = None
    try:
        drv = webdriver.Firefox(options=opts)
        drv.set_page_load_timeout(35)
        drv.get(url)

        # wait for the inner scroll box to show up
        wait = WebDriverWait(drv, 20)
        try:
            container = wait.until(
                EC.presence_of_element_located(
                    (
                        By.CSS_SELECTOR,
                        "div.pd__crossref__list.js-dataContainer.js-infiniteScroll",
                    )
                )
            )
        except TimeoutException:
            container = None
            logger.info("[get_part_models] cross-ref list not found; falling back")

        # preferred: scroll the inner container
        if container:
            models = _scroll_collect_models(drv, container, pause=pause)
        else:
            # fallback: brute-force page scroll
            models = set()
            same_runs, last_h = 0, 0
            while same_runs < 3:
                for a in drv.find_elements(By.CSS_SELECTOR, 'a[href^="/Models/"]'):
                    txt = a.text.strip()
                    if txt.isdigit():
                        models.add(txt)

                drv.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(pause)
                h = drv.execute_script("return document.body.scrollHeight")
                same_runs = same_runs + 1 if h == last_h else 0
                last_h = h

        logger.info(f"[get_part_models] harvested {len(models)} models")
        return sorted(models)

    except Exception as e:
        logger.warning(f"[get_part_models] Selenium error: {e}")
        return []

    finally:
        if drv:
            try:
                drv.quit()
            except Exception:
                pass


# ───────────────────── legacy shim (unchanged) ─────────────────────
def get_parts_models(
    part_code: str,
    *,
    session: requests.Session | None = None,
    pause: float = 1.0,
) -> List[str]:
    return get_part_models(part_code, session=session, pause=pause)


def get_compat(code: str) -> dict:
    kind, _, _ = discover_link(code)
    kind = (kind or "NONE").upper()

    # ───────── MODEL → compatible parts ─────────
    if kind == "MODEL":
        if code not in MODEL_COMPAT:
            logger.info(f"[COMPAT CACHE] populating parts for {code}")
            MODEL_COMPAT[code] = get_model_parts(code)
            _write_cache("model_compat_cache.json", MODEL_COMPAT)
        return {
            "type": "model",
            "model": code,
            "compatible_parts": MODEL_COMPAT[code],
        }

    # ───────── PART → cross‑ref models ─────────
    if kind == "PART":
        models = PART_COMPAT.get(code, [])
        if not models:                                   # ⬅ retry if empty
            models = get_parts_models(code)
            if models:                                   # only persist non‑empty
                PART_COMPAT[code] = models
                _write_cache("part_compat_cache.json", PART_COMPAT)

        # keep session model→part matrix fresh
        for mdl in SESSION_STATE["models"]:
            if mdl not in MODEL_COMPAT or not MODEL_COMPAT[mdl]:
                MODEL_COMPAT[mdl] = get_model_parts(mdl)
                _write_cache("model_compat_cache.json", MODEL_COMPAT)

        comparison = {
            mdl: code in MODEL_COMPAT.get(mdl, []) for mdl in SESSION_STATE["models"]
        }
        return {
            "type": "part",
            "part": code,
            "crossref_models": models,
            "session_models": comparison,
        }

    logger.warning(f"[get_compat] '{code}' could not be resolved as model or part")
    return {"error": "not_resolved"}

# ▲▲▲  END of get_compat() replacement  ▲▲▲
def need_more_context_yes_no(question: str) -> dict:
    """
    Ask the LLM if this query needs more context from previous conversation.
    Returns {"needs_context": "YES"} or {"needs_context": "NO"}
    """
    logger.info(f"[NEED_CONTEXT] Checking if query needs prior context: {question}")
    result = ask_llm(
        system="Does this user query appear to depend on prior context (e.g. previous parts or models mentioned)?",
        user=question
    ).strip().upper()
    logger.info(f"[NEED_CONTEXT] LLM response: {result}")

    if "YES" in result:
        logger.info("[NEED_CONTEXT] Context required.")
        return {"needs_context": "YES"}
    logger.info("[NEED_CONTEXT] No context needed.")
    return {"needs_context": "NO"}


def firecrawl_research(query: str) -> dict:
    logger.info(f"[FIRECRAWL] Researching query: {query}")
    result = research_agent(query)
    logger.info(f"[FIRECRAWL] Result: {str(result)[:300]}...")  # limit for log length
    return {"answer": result}


def fill_context(question: str) -> dict:
    logger.info(f"[CONTEXT] Inferring missing context for: {question}")
    ctx = ask_llm("Briefly infer missing appliance context for this question:", question)
    logger.info(f"[CONTEXT] Filled context: {ctx}")
    return {"context": ctx}

def _compat_wrapper(model_id: str = "", part_id: str = "") -> dict:
    code = model_id or part_id
    logger.info(f"[COMPAT] Checking compatibility for code: {code}")
    result = get_compat(code)
    logger.info(f"[COMPAT] Compatibility result: {str(result)[:300]}...")
    return result

# ─────────────────────────  LANGCHAIN TOOL WRAPPERS  ────────────────
# We wrap existing Python functions so LangGraph can call them automatically.
scrape_model_tool = Tool.from_function(
    name="scrape_model",
    description="Scrape a model page and return visible text and URL.",
    func=scrape_model,
)

scrape_part_tool = Tool.from_function(
    name="scrape_part",
    description="Scrape a part page and return visible text and URL.",
    func=scrape_part,
)

get_compat_tool = Tool.from_function(
    name="get_compat",
    description="Check compatibility: if given a model, return its parts; if given a part, return model cross‑refs.",
    func=_compat_wrapper,
)

firecrawl_research_tool = Tool.from_function(
    name="firecrawl_research",
    description=(
    "Use for general open-ended questions that aren't tied to specific parts or models, "
    "especially if no relevant result was found by other tools. "
    "Fallback research tool when structured answers fail."),
    func=lambda query: {"answer": research_agent(query)},
)

need_context_tool = Tool.from_function(
    name="need_more_context_yes_no",
    description=(
    "Decides if the user’s query needs prior context to be understood (e.g., "
    "mentions 'this model' or 'that part' without naming it). "
    "Returns YES or NO."),
    func=need_more_context_yes_no
)

fill_context_tool = Tool.from_function(
    name="fill_context",
    description=(
        "Try to infer the missing appliance context from a vague or incomplete user question. "
        "Use when the user refers to something like 'this model' or 'that part' without naming it, "
        "and no clear context is available. Outputs a short guess about what appliance or part they're asking about."
    ),
    func=lambda question: {
        "context": ask_llm(
            "Briefly infer missing appliance context for this question:", question
        )
    },
)

TOOLS_LIST = [
    scrape_model_tool,
    scrape_part_tool,
    get_compat_tool,
    firecrawl_research_tool,
    fill_context_tool,
    need_context_tool
]

# ───────────────────────  LANGGRAPH AGENT SETUP  ─────────────────────
LC_MODEL = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

AGENT = create_react_agent(LC_MODEL, TOOLS_LIST)

# System prompt guiding the agent’s behaviour
AGENT_SYS_PROMPT = (
    "You are a tool‑planning assistant for appliance parts. "
    "Think step‑by‑step, decide which tool to call, and keep calling tools "
    "until you can answer the user. Use JSON tool calls as needed."
)

# ───────────────────────  ANALYSIS PRE‑PROCESS  ─────────────────────
def analyse(q: str) -> dict:
    logger.info(f"[ANALYSE] Original question: {q}")
    compat_intent = llm_is_compat(q) == "YES"
    tokens = extract_model_numbers(q)
    models, parts = [], []
    for tok in tokens:
        code_type = lookup_code_type(tok)
        if code_type == "MODEL": models.append(tok.upper())
        elif code_type == "PART": parts.append(tok.upper())
    models, parts = fallback_codes(models, parts, q)
    update_session(models, parts)
    return {"compat_intent": compat_intent, "models": models, "parts": parts}

# ───────────────────────  ORCHESTRATOR LOOP  ────────────────────────
def process_question(q: str) -> str:
    logger.info(f"[DEBUG] incoming: {q!r}")
    analysis = analyse(q)

    # Pre‑warm compatibility cache if needed
    if analysis["compat_intent"]:
        for mdl in SESSION_STATE["models"]:
            if mdl not in MODEL_COMPAT or not MODEL_COMPAT[mdl]:
                logger.info(f"[COMPAT CACHE] populating parts for {mdl}")
                MODEL_COMPAT[mdl] = get_model_parts(mdl)
                _write_cache(COMPAT_CACHE_FILE, MODEL_COMPAT)

    # Build conversation context for agent
    messages = [
        {"role": "system", "content": AGENT_SYS_PROMPT},
        {"role": "system", "content": f"analysis: {analysis}"},
        {"role": "user",   "content": q},
    ]

    try:
        agent_response = AGENT.invoke({"messages": messages})
        for step in agent_response.get("intermediate_steps", []):
            logger.info(f"[TOOL CALL DEBUG] Tool used: {step.tool.name}")
            logger.info(f"[TOOL CALL DEBUG] Arguments: {step.tool_input}")
            logger.info(f"[TOOL CALL DEBUG] Output: {step.output}")

        ai_message = agent_response["messages"][-1].content
        logger.info(f"[DEBUG] FINAL ANSWER (LangGraph): {ai_message}")
        return ai_message
    except Exception as e:
        logger.exception(e)
        return "Sorry, I ran into a problem answering that."

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
